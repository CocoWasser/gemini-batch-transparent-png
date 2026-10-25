"""Create local checkerboard contact sheets and advisory quality reports."""

import csv
import io
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


THUMBNAIL = 240
COLUMNS = 4
PAGE_SIZE = 16
LABEL_FONT = ImageFont.load_default(size=14)
NOTE_FONT = ImageFont.load_default(size=12)


def checkerboard(image, size=None, tile=16):
    """Composite an RGBA image over a neutral checkerboard for inspection."""
    foreground = image.convert("RGBA")
    if size is not None:
        foreground.thumbnail(size, Image.Resampling.LANCZOS)
    canvas_size = size or foreground.size
    canvas = Image.new("RGBA", canvas_size, (192, 192, 192, 255))
    draw = ImageDraw.Draw(canvas)
    for y in range(0, canvas.height, tile):
        for x in range(0, canvas.width, tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(236, 236, 236, 255))
    offset = ((canvas.width - foreground.width) // 2, (canvas.height - foreground.height) // 2)
    canvas.alpha_composite(foreground, offset)
    return canvas.convert("RGB")


def assess_quality(png, white_file=None, black_file=None):
    """Flag suspicious pixels for review; never treat these as segmentation truth.

    Matching near-white opaque pixels can be unedited background holes OR valid
    white foreground details. Color-difference disagreement can indicate edits
    to the subject. These signals must not trigger automatic paid retries.
    """
    with Image.open(png) as image:
        if image.mode != "RGBA":
            raise ValueError("Expected an RGBA image")
        rgba = np.asarray(image, dtype=np.uint8)
    alpha = rgba[:, :, 3]
    near_white = np.all(rgba[:, :, :3] >= 238, axis=2) & (alpha >= 244)
    edge = max(1, min(alpha.shape) // 100)
    border = np.concatenate([
        alpha[:edge].ravel(), alpha[-edge:].ravel(),
        alpha[:, :edge].ravel(), alpha[:, -edge:].ravel(),
    ])
    opaque_white_pct = 100 * float(near_white.mean())
    edge_pct = 100 * float((border > 25).mean())
    channel_pct = None
    flags = []
    if opaque_white_pct > 0.5:
        flags.append("check near-white opaque areas")
    if edge_pct > 3:
        flags.append("nontransparent canvas edge")
    if not np.any(alpha > 0):
        flags.append("fully transparent image")
    elif np.all(alpha > 250):
        flags.append("no transparent background")

    if white_file is not None and black_file is not None:
        if white_file.exists() and black_file.exists():
            with Image.open(white_file) as image:
                white = np.asarray(image.convert("RGB"), dtype=np.float32)
            with Image.open(black_file) as image:
                black = np.asarray(image.convert("RGB"), dtype=np.float32)
            if white.shape != black.shape or white.shape[:2] != alpha.shape:
                flags.append("source dimensions do not match")
            else:
                delta = white - black
                inconsistent = (np.ptp(delta, axis=2) > 30) | (np.min(delta, axis=2) < -15)
                channel_pct = 100 * float(inconsistent.mean())
                if channel_pct > 1:
                    flags.append("foreground/background pair differs")
        else:
            flags.append("source pair unavailable")

    return {
        "near_white_opaque_pct": round(opaque_white_pct, 3),
        "nontransparent_edge_pct": round(edge_pct, 3),
        "inconsistent_pair_pct": "" if channel_pct is None else round(channel_pct, 3),
        "review_notes": "; ".join(flags),
    }


def _save_image(image, target):
    temporary = target.with_name(target.name + ".tmp")
    image.save(temporary, format="JPEG", quality=90)
    os.replace(temporary, target)


def _wrapped_lines(text, width, font):
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def write_review(image_ids, base, *, name="overview", variant="transparent", pair_base=None,
                 page_size=PAGE_SIZE):
    """Write bounded contact sheets and a TSV, including missing/invalid entries.

    Each page is rendered independently to keep memory bounded for large jobs.
    The caller holds the output lock. Existing pages for this exact name are
    replaced, and obsolete pages are removed after successful rendering.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Review name must contain only letters, digits, underscores, or hyphens")
    if not 1 <= page_size <= 64:
        raise ValueError("Review page size must be between 1 and 64")
    ids = list(dict.fromkeys(image_ids))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", image_id) for image_id in ids):
        raise ValueError("Invalid image ID in review")
    if not ids:
        return []
    base = Path(base)
    pair_base = Path(pair_base) if pair_base is not None else base
    output = base / "review"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pages = []
    margin, cell_height = 16, THUMBNAIL + 58
    for page_start in range(0, len(ids), page_size):
        page_ids = ids[page_start:page_start + page_size]
        columns = min(COLUMNS, len(page_ids))
        width = margin + columns * (THUMBNAIL + margin)
        heading = _wrapped_lines(
            f"{name} | page {len(pages) + 1} | checkerboard alpha review", width - 2 * margin, LABEL_FONT)
        notes = _wrapped_lines(
            "Flags are review hints. White foreground details may be legitimate.", width - 2 * margin, NOTE_FONT)
        header = 24 + 19 * len(heading) + 17 * len(notes)
        height = header + math.ceil(len(page_ids) / columns) * cell_height + margin
        sheet = Image.new("RGB", (width, height), "#f4f5f7")
        draw = ImageDraw.Draw(sheet)
        text_y = 12
        for line in heading:
            draw.text((margin, text_y), line, fill="#182230", font=LABEL_FONT)
            text_y += 19
        for line in notes:
            draw.text((margin, text_y), line, fill="#475467", font=NOTE_FONT)
            text_y += 17
        for index, image_id in enumerate(page_ids):
            png = base / variant / f"{image_id}.png"
            row = {"id": image_id, "status": "pending", "near_white_opaque_pct": "",
                   "nontransparent_edge_pct": "", "inconsistent_pair_pct": "", "review_notes": "image not available"}
            preview = Image.new("RGB", (THUMBNAIL, THUMBNAIL), "#e4e7ec")
            if png.exists():
                try:
                    row.update(assess_quality(png, pair_base / "white" / png.name,
                                              pair_base / "black" / png.name))
                    with Image.open(png) as image:
                        preview = checkerboard(image, (THUMBNAIL, THUMBNAIL))
                    row["status"] = "review" if row["review_notes"] else "ready"
                except (OSError, ValueError) as exc:
                    row.update(status="invalid", review_notes=str(exc))
            rows.append(row)
            x = margin + (index % columns) * (THUMBNAIL + margin)
            y = header + (index // columns) * cell_height
            sheet.paste(preview, (x, y))
            label = image_id
            if LABEL_FONT.getlength(label) > THUMBNAIL:
                while LABEL_FONT.getlength(label + "...") > THUMBNAIL:
                    label = label[:-1]
                label += "..."
            draw.text((x, y + THUMBNAIL + 8), label, fill="#182230", font=LABEL_FONT)
            color = "#b54708" if row["status"] != "ready" else "#067647"
            draw.text((x, y + THUMBNAIL + 30), row["status"].upper(), fill=color, font=NOTE_FONT)
            if row["status"] == "review":
                draw.text((x + 60, y + THUMBNAIL + 30), "see TSV notes", fill="#475467", font=NOTE_FONT)
        target = output / f"{name}-{len(pages) + 1:03d}.jpg"
        _save_image(sheet, target)
        pages.append(target)

    report = output / f"{name}.tsv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(buffer.getvalue(), encoding="utf-8")
    os.replace(temporary, report)
    for obsolete in output.glob(f"{name}-*.jpg"):
        if re.fullmatch(re.escape(name) + r"-\d{3,}\.jpg", obsolete.name) and obsolete not in pages:
            obsolete.unlink()
    print(f"Review sheets: {output / (name + '-*.jpg')}")
    print(f"Quality report: {report}")
    return pages
