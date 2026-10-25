#!/usr/bin/env python3
"""Generate a public example with two synchronous Gemini requests, then export it.

Use --from-output to export a completed Batch run without any API calls.
"""

import argparse
import base64
import getpass
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import batch_transparent as bt
from review import checkerboard


def generate_image(parts, model, size, api_key):
    """Use the same REST request schema as the Batch generator."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"
    with requests.post(
        url, headers={"x-goog-api-key": api_key}, timeout=(30, 300),
        json={"contents": [{"role": "user", "parts": parts}],
              "generationConfig": {"responseModalities": ["IMAGE"],
                                   "imageConfig": {"aspectRatio": "1:1", "imageSize": size}}},
    ) as response:
        payload = response.json()
        if not response.ok:
            message = str(payload.get("error", {}).get("message", "Request failed"))
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {message.replace(api_key, '[redacted]')}")
    image = bt.extract_image(payload)
    if image is None:
        raise ValueError("Gemini did not return an image")
    return image


def export_comparison(base, image_id, destination):
    source_paths = {stage: base / stage / f"{image_id}.png"
                    for stage in ("white", "black", "transparent")}
    images = {}
    for stage, path in source_paths.items():
        with Image.open(path) as image:
            images[stage] = image.convert("RGBA" if stage == "transparent" else "RGB")
    if len({image.size for image in images.values()}) != 1:
        raise ValueError("Demo image dimensions do not match")
    destination.mkdir(parents=True, exist_ok=True)
    width, margin, top = 400, 24, 68
    sheet = Image.new("RGB", (3 * width + 4 * margin, width + top + margin), "#f4f5f7")
    draw = ImageDraw.Draw(sheet)
    titles = {"white": "1  WHITE BACKGROUND", "black": "2  BLACK BACKGROUND",
              "transparent": "3  RECONSTRUCTED ALPHA"}
    for index, (stage, image) in enumerate(images.items()):
        image.save(destination / f"{stage}.png")
        x = margin + index * (width + margin)
        draw.text((x, 24), titles[stage], fill="#182230", font=ImageFont.load_default(size=18))
        if stage == "transparent":
            thumb = checkerboard(image, (width, width), tile=20)
        else:
            thumb = Image.new("RGB", (width, width), "white" if stage == "white" else "black")
            fitted = image.copy()
            fitted.thumbnail((width, width), Image.Resampling.LANCZOS)
            thumb.paste(fitted, ((width-fitted.width)//2, (width-fitted.height)//2))
        sheet.paste(thumb, (x, top))
    sheet.save(destination / "comparison.jpg", quality=94)
    print(f"Published demo images to {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=Path("example.tsv"))
    parser.add_argument("--id", default="demo_01")
    parser.add_argument("--out", type=Path, default=Path("output_demo"))
    parser.add_argument("--publish-to", type=Path, default=Path("docs/images"))
    parser.add_argument("--from-output", type=Path, help="Reuse completed Batch images; no API calls")
    parser.add_argument("--model", default=bt.DEFAULT_MODEL)
    parser.add_argument("--size", choices=["1K", "2K", "4K"], default="1K")
    args = parser.parse_args(argv)
    records = {row[0]: row for row in bt.load_tsv(args.tsv)}
    if args.id not in records:
        parser.error("The selected ID is not in the TSV")
    base = (args.from_output or args.out).resolve()
    if args.from_output is not None:
        with bt.output_lock(base):
            export_comparison(base, args.id, args.publish_to)
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and sys.stdin.isatty():
        api_key = getpass.getpass("Gemini API key (hidden, not saved): ")
    if not api_key:
        parser.error("Set GEMINI_API_KEY before generating the demo")
    for stage in ("white", "black", "transparent", "_work"):
        (base / stage).mkdir(parents=True, exist_ok=True)
    with bt.output_lock(base):
        manifest_path = base / "_work" / "demo.json"
        manifest = {"record": list(records[args.id]), "model": args.model, "size": args.size}
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Demo settings changed; choose a new --out directory")
        if not manifest_path.exists():
            bt.atomic_bytes(manifest_path, json.dumps(manifest).encode("utf-8"))
        for stage in ("white", "black"):
            target = base / stage / f"{args.id}.png"
            if target.exists():
                with Image.open(target) as image:
                    bt.validate_background(image, stage)
                print(f"Reusing {target}", flush=True)
                continue
            if stage == "white":
                parts = [{"text": bt.white_prompt(records[args.id][2])}]
            else:
                data = (base / "white" / f"{args.id}.png").read_bytes()
                parts = [{"text": bt.BLACK_PROMPT}, {"inlineData": {
                    "data": base64.b64encode(data).decode("ascii"), "mimeType": "image/png"}}]
            print(f"Generating {stage} background with {args.model}...", flush=True)
            image = generate_image(parts, args.model, args.size, api_key)
            bt.validate_background(image, stage)
            image.save(target)
        bt.compose_transparent(base / "white" / f"{args.id}.png",
                               base / "black" / f"{args.id}.png",
                               base / "transparent" / f"{args.id}.png")
        bt.save_review([args.id], base)
        export_comparison(base, args.id, args.publish_to)


if __name__ == "__main__":
    main()
