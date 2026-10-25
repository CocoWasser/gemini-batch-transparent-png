#!/usr/bin/env python3
"""Review, regenerate, and repair selected transparent images.

Examples:
  python repair.py local demo_01 demo_02
  python repair.py regen demo_01 --tsv example.tsv
  python repair.py retry demo_01
  python repair.py accept regen demo_01
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import batch_transparent as bt


STRICT_BLACK_PROMPT = """
This is a precise BACKGROUND REPLACEMENT task, not a new drawing.
Use the supplied white-background image as the exact source.
Replace ALL of the white canvas BACKGROUND with uniform pure RGB black #000000.
This includes every visible area of empty canvas inside and around the subject:
gaps between parts, negative space behind the subject, and any enclosed holes.
There must be no white, off-white, gray, or colored BACKGROUND patches anywhere.
Do not turn the foreground subject into a black silhouette.
White, cream, and pale-colored details that belong to the foreground must remain
unchanged; distinguish them from empty canvas by their outlines and structure.
Keep every foreground pixel, color, outline, size, location, and composition
identical to the input. Do not redraw, move, resize, recolor or add anything.
No checkerboard, no shadow, no gradient. Output a fully opaque RGB image.
"""


def parse_ids(raw):
    values = [value.strip() for value in raw]
    if not values or any(not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in values):
        raise ValueError("IDs may contain only ASCII letters, digits, underscores, and hyphens")
    return list(dict.fromkeys(values))


def read_state(base):
    path = base / "_work" / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def has_pending_batch(state):
    return bool(state.get("current") or any(state.get("jobs", {}).values()))


def backup_directory(base, category):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / "_work" / category / f"{stamp}_{uuid.uuid4().hex[:8]}"


def corners(img):
    height, width = img.shape[:2]
    patch = max(4, min(height, width) // 64)
    return np.concatenate([
        img[:patch, :patch].reshape(-1, 3),
        img[:patch, -patch:].reshape(-1, 3),
        img[-patch:, :patch].reshape(-1, 3),
        img[-patch:, -patch:].reshape(-1, 3),
    ], axis=0)


def recomposite(white_file, black_file, target):
    with Image.open(white_file) as image:
        white = np.asarray(image.convert("RGB"), dtype=np.float32)
    with Image.open(black_file) as image:
        black = np.asarray(image.convert("RGB"), dtype=np.float32)
    if white.shape != black.shape:
        raise bt.ImageSizeMismatch("White and black image dimensions do not match")

    white_bg = np.median(corners(white), axis=0)
    black_bg = np.median(corners(black), axis=0)
    bg_delta = white_bg - black_bg
    if np.min(bg_delta) < 180:
        raise ValueError(
            f"Insufficient background contrast at the corners: white={white_bg.round()}, black={black_bg.round()}"
        )

    # Estimate transparency from the measured backgrounds of both images.
    trans = np.median((white - black) / bg_delta, axis=2)
    alpha = 1.0 - np.clip(trans, 0, 1)
    same_as_background = (
        (np.max(np.abs(white - white_bg), axis=2) < 12)
        & (np.max(np.abs(black - black_bg), axis=2) < 12)
        & (alpha < 0.10)
    )
    alpha[same_as_background] = 0
    alpha[alpha < 0.01] = 0

    aa = alpha[..., None]
    safe = np.maximum(aa, 0.02)
    fore_w = (white - (1 - aa) * white_bg) / safe
    fore_b = (black - (1 - aa) * black_bg) / safe
    color = np.clip((fore_w + fore_b) / 2, 0, 255)
    color[alpha == 0] = 0
    rgba = np.dstack((
        np.rint(color).astype("uint8"),
        np.rint(alpha * 255).astype("uint8"),
    ))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.png")
    Image.fromarray(rgba).save(temporary, format="PNG")
    os.replace(temporary, target)

    pale = np.all(rgba[:, :, :3] >= 238, axis=2) & (rgba[:, :, 3] >= 244)
    edges = np.concatenate([
        alpha[:8, :].ravel(), alpha[-8:, :].ravel(),
        alpha[:, :8].ravel(), alpha[:, -8:].ravel(),
    ])
    return 100 * float(pale.mean()), 100 * float(np.mean(edges > 0.10))


def create_preview(png, dest):
    with Image.open(png) as image:
        foreground = image.convert("RGBA")
    width, height = foreground.size
    base = Image.new("RGBA", (width, height), (190, 190, 190, 255))
    draw = ImageDraw.Draw(base)
    tile = 32
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle(
                    (x, y, x + tile - 1, y + tile - 1),
                    fill=(235, 235, 235, 255),
                )
    base.alpha_composite(foreground)
    dest.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(dest)


def run_local(ids, white_dir, black_dir, repair, variant):
    count = 0
    for image_id in ids:
        white = white_dir / f"{image_id}.png"
        black = black_dir / f"{image_id}.png"
        if not white.exists() or not black.exists():
            print(f"{image_id}: missing white or black image; skipped")
            continue
        try:
            candidate = repair / variant / f"{image_id}.png"
            pale, edge = recomposite(white, black, candidate)
            create_preview(
                candidate, repair / f"{variant}_preview" / f"{image_id}.jpg"
            )
            warning = "  [manual review recommended]" if pale > 18 or edge > 3 else ""
            print(
                f"{image_id}: candidate saved; near-white opaque pixels {pale:.1f}%, "
                f"nontransparent edge pixels {edge:.1f}%{warning}"
            )
            count += 1
        except (OSError, ValueError) as exc:
            print(f"{image_id}: local composition failed: {exc}")
    print(f"Saved {count}/{len(ids)} candidates: {repair / variant}")
    print(f"Previews: {repair / (variant + '_preview')}")
    bt.save_review(ids, repair, name=f"{variant}-overview", variant=variant,
                   pair_base=white_dir.parent)
    return count == len(ids)


def regen(ids, source, repair, args):
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is not set")

    lookup = {row[0]: row for row in bt.load_tsv(args.tsv)}
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(f"IDs missing from the TSV: {', '.join(sorted(missing))}")
    records = [lookup[image_id] for image_id in ids]
    source_state = read_state(source)
    repair_state = read_state(repair)

    # Preserve the model and resolution when resuming a repair job.
    args.model = (args.model or repair_state.get("model")
                  or source_state.get("model") or bt.DEFAULT_MODEL)
    args.size = (args.size or repair_state.get("size")
                 or source_state.get("size") or "1K")
    current = repair_state.get("current") or {}
    if set(current.get("keys", [])) - set(ids):
        raise ValueError("An unfinished batch contains other IDs; resume it with the original IDs first")
    repair_tsv = repair / "repair.tsv"
    if any(repair_state.get("jobs", {}).values()):
        if not repair_tsv.exists() or bt.load_tsv(repair_tsv) != records:
            raise ValueError("An unfinished legacy batch exists; resume it with the original repair TSV and IDs")

    # Validate all inputs before writing state or copying any files.
    whites = {}
    for image_id in ids:
        candidate = repair / "white" / f"{image_id}.png"
        original = source / "white" / f"{image_id}.png"
        white = candidate if candidate.exists() else original
        if not white.exists():
            raise ValueError(f"Missing white-background source image: {original}")
        with Image.open(white) as image:
            bt.validate_background(image, "white")
        whites[image_id] = white

    for folder in ("white", "black", "transparent", "_work"):
        (repair / folder).mkdir(parents=True, exist_ok=True)
    bt.initialize_state(records, repair / "_work", args.model, args.size)
    for image_id, white in whites.items():
        dest = repair / "white" / f"{image_id}.png"
        if white != dest:
            bt.atomic_bytes(dest, white.read_bytes())

    # Keep the selected prompts local for inspection and later retries.
    import csv
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(["id", "name", "prompt"])
    writer.writerows(records)
    bt.atomic_bytes(repair_tsv, buffer.getvalue().encode("utf-8"))

    # Reuse the Batch runner under the locks already held by this command.
    bt.run_batch(records, repair, args, STRICT_BLACK_PROMPT)
    return run_local(ids, repair / "white", repair / "black", repair, "regen")


def retry(ids, repair):
    state_file = repair / "_work" / "state.json"
    if not state_file.exists():
        raise ValueError("No repair state file was found")
    state = read_state(repair)
    if has_pending_batch(state):
        raise ValueError("An unfinished batch exists; resume it with the original IDs before resetting attempts")
    unknown = set(ids) - state.get("fingerprints", {}).keys()
    if unknown:
        raise ValueError(f"These IDs are not in the repair job: {', '.join(sorted(unknown))}")

    backup = backup_directory(repair, "manual_rejections")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state_file, backup / "state.json")
    for image_id in ids:
        for sub in ("black", "transparent", "regen", "regen_preview"):
            ext = ".jpg" if sub.endswith("preview") else ".png"
            file = repair / sub / (image_id + ext)
            if file.exists():
                target = backup / sub / (image_id + ext)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(target))
        state.setdefault("attempts", {}).setdefault("black", {}).pop(image_id, None)
        state.setdefault("errors", {}).setdefault("black", {}).pop(image_id, None)
    bt.save_state(state, repair / "_work")
    print(f"Previous candidates and state were backed up to {backup}")
    print("Run regen again with the same IDs, TSV, and directories. Generation will call the API.")


def accept(ids, variant, source, repair):
    if has_pending_batch(read_state(source)) or has_pending_batch(read_state(repair)):
        raise ValueError("An unfinished batch exists; resume it before accepting candidates")
    # Validate the entire selection before replacing any accepted images.
    for image_id in ids:
        candidate = repair / variant / f"{image_id}.png"
        with Image.open(candidate) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGBA":
                raise ValueError(f"The candidate must be an RGBA PNG: {candidate}")

    backup = backup_directory(source, "quality_backup")
    destination_dir = source / "transparent"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        candidate = repair / variant / f"{image_id}.png"
        destination = destination_dir / f"{image_id}.png"
        if destination.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup / f"{image_id}.png")
        bt.atomic_bytes(destination, candidate.read_bytes())
        print(f"Replaced: {destination}")
    if backup.exists():
        print(f"Original image backups: {backup}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Repair selected transparent PNGs, preview candidates, and accept them manually."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=Path("output"),
                        help="Source batch directory (default: output)")
    common.add_argument("--repair", type=Path, default=Path("output_repair"),
                        help="Repair directory (default: output_repair)")
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("local", parents=[common], help="Reconstruct alpha from existing white/black pairs")
    local.add_argument("ids", nargs="+")
    generate = sub.add_parser("regen", parents=[common], help="Reuse white images and regenerate black backgrounds")
    generate.add_argument("ids", nargs="+")
    generate.add_argument("--tsv", type=Path, default=Path("example.tsv"))
    generate.add_argument("--model", help="Inherit the repair or source model by default")
    generate.add_argument("--size", choices=["1K", "2K", "4K"],
                          help="Inherit the repair or source resolution by default")
    generate.add_argument("--batch-size", type=int, default=20)
    generate.add_argument("--max-attempts", type=int, default=bt.DEFAULT_MAX_ATTEMPTS)
    generate.add_argument("--confirm-not-created", action="store_true",
                          help="Use only after manually confirming the job was not created on the server")
    reject = sub.add_parser("retry", parents=[common], help="Back up rejected candidates and reset attempts")
    reject.add_argument("ids", nargs="+")
    approve = sub.add_parser("accept", parents=[common], help="Back up original images and accept candidates")
    approve.add_argument("variant", choices=["local", "regen"])
    approve.add_argument("ids", nargs="+")
    args = parser.parse_args(argv)

    try:
        ids = parse_ids(args.ids)
        source, repair = args.source.resolve(), args.repair.resolve()
        if source == repair or source in repair.parents or repair in source.parents:
            raise ValueError("Source and repair directories must be separate and must not be nested")
        if args.command == "regen":
            if not 1 <= args.batch_size <= 100 or args.max_attempts < 1:
                raise ValueError("--batch-size must be between 1 and 100; --max-attempts must be positive")

        with ExitStack() as stack:
            for directory in sorted({source, repair}):
                stack.enter_context(bt.output_lock(directory))
            if args.command == "local":
                success = run_local(ids, source / "white", source / "black", repair, "local")
                return 0 if success else 1
            if args.command == "regen":
                return 0 if regen(ids, source, repair, args) else 1
            if args.command == "retry":
                retry(ids, repair)
            elif args.command == "accept":
                accept(ids, args.variant, source, repair)
        return 0
    except (OSError, ValueError, RuntimeError, bt.StopRun) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
