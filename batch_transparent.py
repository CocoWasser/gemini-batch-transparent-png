#!/usr/bin/env python3
"""Generate transparent PNGs with Gemini Batch and paired background matting."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import time
import uuid
import fcntl
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from google import genai
from google.genai import types

from review import write_review


# ==================== Configuration ====================

DEFAULT_MODEL = "gemini-nano-banana-2.1"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_ATTEMPTS = 3
POLL_SECONDS = 60

BLACK_PROMPT = """
Edit the provided image.

Change ONLY the solid white background to perfectly
uniform pure black (#000000).

Keep the foreground subject exactly the same:
- Identical shape and silhouette
- Identical position, scale and orientation
- Identical black outlines and colors
- Identical composition and canvas dimensions
- Preserve edge antialiasing against the new background

Do not redraw, redesign, shift, resize or recolor
the subject. Do not introduce shadows or gradients.

The background must be pure RGB black (#000000),
not transparent, and not a checkerboard.
"""

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class StopRun(Exception):
    """Pause safely while preserving the state file for a later resume."""
    pass


class ImageSizeMismatch(ValueError):
    """The white and black source images have different dimensions."""


def is_quota_error(error):
    code = getattr(error, "code", None)
    message = str(error).upper()

    return (
        str(code) == "429"
        or "RESOURCE_EXHAUSTED" in message
        or "QUOTA_EXCEEDED" in message
        or "RATE LIMIT" in message
        or re.search(r"\b429\b", message) is not None
    )


def save_state(state, work):
    target = work / "state.json"
    temp = work / "state.json.tmp"

    with temp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, target)


def atomic_bytes(path, data):
    temp = path.with_name(path.name + ".tmp")

    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)


@contextmanager
def output_lock(base):
    """Protect an output directory from simultaneous writers (macOS/Linux)."""
    work = base / "_work"
    work.mkdir(parents=True, exist_ok=True)
    with (work / "run.lock").open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using this directory: {base}") from exc
        yield


def image_path(base, stage, image_id):
    return base / stage / (image_id + ".png")


# ==================== TSV ====================

def load_tsv(path, limit=None):
    records = []
    seen = set()
    first_content_row = True
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")

        for line_number, columns in enumerate(reader, 1):
            if not columns or not any(x.strip() for x in columns):
                continue

            if len(columns) < 3:
                raise ValueError(
                    f"Row {line_number} must contain at least three columns"
                )

            image_id = columns[0].strip()
            name = columns[1].strip()
            prompt = "\t".join(columns[2:]).strip()

            is_header = first_content_row and image_id.lower() in (
                "id", "\u7f16\u53f7"
            )
            first_content_row = False
            if is_header:
                continue

            if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
                raise ValueError(f"Invalid ID: {image_id}")

            if image_id in seen:
                raise ValueError(f"Duplicate ID: {image_id}")

            if not prompt:
                raise ValueError(f"{image_id} has an empty prompt")

            seen.add(image_id)
            records.append((image_id, name, prompt))

    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError("The TSV contains no valid records")

    return records


def white_prompt(original):
    text = re.sub(
        r"isolated on (?:a )?transparent background",
        "isolated on a pure white (#FFFFFF) background",
        original,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"transparent background",
        "pure white (#FFFFFF) background",
        text,
        flags=re.IGNORECASE,
    )

    return text + """

IMPORTANT:
Generate an opaque RGB image with an exactly uniform
pure white (#FFFFFF) background.
No transparency, no checkerboard, no cast shadows.
Do not add text, labels, frames or decorations.
Keep the entire subject inside the square canvas.
"""


# ==================== State management ====================

def initialize_state(records, work, model, size):
    state_file = work / "state.json"

    if state_file.exists():
        state = json.loads(
            state_file.read_text(encoding="utf-8")
        )
    else:
        state = {
            "model": model,
            "size": size,
            "fingerprints": {},
            "jobs": {},
        }

    if state.get("model") != model:
        raise ValueError(
            "Model differs from the saved job; use the original model or a new output directory"
        )

    if state.get("size") != size:
        raise ValueError(
            "Resolution differs from the saved job; use a new output directory"
        )

    state.setdefault("fingerprints", {})
    state.setdefault("jobs", {})
    state.setdefault("current", None)
    state.setdefault("attempts", {"white": {}, "black": {}})
    state.setdefault("errors", {"white": {}, "black": {}})

    state["attempts"].setdefault("white", {})
    state["attempts"].setdefault("black", {})
    state["errors"].setdefault("white", {})
    state["errors"].setdefault("black", {})

    for image_id, name, prompt in records:
        fingerprint = hashlib.sha256(
            (name + "\0" + prompt).encode("utf-8")
        ).hexdigest()

        old = state["fingerprints"].get(image_id)

        if old and old != fingerprint:
            raise ValueError(
                f"The name or prompt for {image_id} has changed. "
                "Use a new output directory to avoid mixing images."
            )

        state["fingerprints"][image_id] = fingerprint

    # Recover pending jobs stored in the legacy state.json format.
    if state["current"] is None:
        legacy_jobs = [
            (stage, name)
            for stage, name in state["jobs"].items()
            if name and stage in ("white", "black")
        ]

        if len(legacy_jobs) > 1:
            raise StopRun(
                "The legacy state contains multiple unfinished batches. "
                "Review them manually; no duplicate requests will be submitted."
            )

        if legacy_jobs:
            stage, job_name = legacy_jobs[0]
            request_file = work / f"{stage}_requests.jsonl"

            if not request_file.exists():
                raise StopRun(
                    f"Legacy job {job_name} exists, but its request file is missing:"
                    f" {request_file}. Keep the state file and inspect the job."
                )

            keys = []

            for line in request_file.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.strip():
                    keys.append(json.loads(line)["key"])

            state["current"] = {
                "token": "legacy-" + stage,
                "stage": stage,
                "keys": keys,
                "jsonl": str(request_file),
                "display_name": None,
                "status": "submitted",
                "job_name": job_name,
                "upload_name": None,
                "counted": False,
            }

            state["jobs"] = {}

            print(f"Recovered legacy job: {job_name}")

    save_state(state, work)
    return state


def mark_attempts(state, current, work):
    if current.get("counted"):
        return

    stage = current["stage"]

    for image_id in current["keys"]:
        old = state["attempts"][stage].get(image_id, 0)
        state["attempts"][stage][image_id] = old + 1

    current["counted"] = True
    save_state(state, work)


# ==================== Image quality checks ====================

def transparent_output_error(path):
    """Return a structural error, or None for a fully decodable RGBA PNG.

    Pixel-level quality flags belong to review.py and must not trigger paid
    retries. Verify PNG integrity as well as decoding: opening only the header
    can miss truncated image data or damaged chunks.
    """
    try:
        with open(path, "rb") as source:
            with Image.open(source) as image:
                if image.format != "PNG" or image.mode != "RGBA":
                    return "Expected an RGBA PNG"
                image.verify()

            # Pillow stops verification at the IEND header, before its CRC.
            # Require the complete terminal chunk, including length and CRC.
            source.seek(-12, os.SEEK_END)
            if source.read(12) != b"\x00\x00\x00\x00IEND\xaeB\x60\x82":
                return "Missing or damaged PNG IEND chunk"

        # verify() does not decode pixels and leaves the image unusable.
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        return str(exc)
    return None


def reuse_transparent(base, image_id):
    """Preserve valid outputs; quarantine invalid ones before reconstruction."""
    path = image_path(base, "transparent", image_id)
    if not path.exists():
        return False
    error = transparent_output_error(path)
    if error is None:
        return True
    quarantine(base, "transparent", image_id, error)
    return False


def validate_background(image, stage):
    if stage not in ("white", "black"):
        raise ValueError("Background stage must be white or black")
    if ("A" in image.getbands() or "transparency" in image.info) and (
        image.convert("RGBA").getchannel("A").getextrema()[0] < 255
    ):
        raise ValueError("Source images must be opaque")
    arr = np.asarray(
        image.convert("RGB"), dtype=np.uint8
    )

    height, width = arr.shape[:2]

    if height < 64 or width < 64:
        raise ValueError("Image dimensions must be at least 64 x 64 pixels")

    patch = max(2, min(height, width) // 50)
    band = max(1, min(height, width) // 100)

    corners = [
        arr[:patch, :patch],
        arr[:patch, -patch:],
        arr[-patch:, :patch],
        arr[-patch:, -patch:],
    ]

    if stage == "black":
        def background_pixels(pixels):
            return np.all(pixels <= 30, axis=-1)
    else:
        def background_pixels(pixels):
            return np.all(pixels >= 230, axis=-1)

    good_corners = sum(
        float(np.mean(background_pixels(c))) >= 0.85
        for c in corners
    )

    edge_pixels = np.concatenate([
        arr[:band, :, :].reshape(-1, 3),
        arr[-band:, :, :].reshape(-1, 3),
        arr[:, :band, :].reshape(-1, 3),
        arr[:, -band:, :].reshape(-1, 3),
    ])

    edge_ratio = float(
        np.mean(background_pixels(edge_pixels))
    )

    if good_corners < 3 or edge_ratio < 0.80:
        color = "black" if stage == "black" else "white"

        raise ValueError(
            f"Invalid solid {color} background: "
            f"{good_corners}/4 corners passed; "
            f"background edge coverage {edge_ratio:.1%}"
        )


def quarantine(base, stage, image_id, reason):
    source = image_path(base, stage, image_id)

    if not source.exists():
        return

    rejected = base / "_work" / "rejected" / stage
    rejected.mkdir(parents=True, exist_ok=True)

    destination = rejected / (
        image_id + "-" + uuid.uuid4().hex[:8] + ".png"
    )

    os.replace(source, destination)

    print(f"  Quarantined {stage}/{image_id}.png: {reason}")


def check_existing(records, base, state, work):
    changed = False

    for image_id, name, prompt in records:
        if reuse_transparent(base, image_id):
            continue

        white_file = image_path(base, "white", image_id)
        black_file = image_path(base, "black", image_id)

        # Validate the white background.
        if white_file.exists():
            try:
                with Image.open(white_file) as image:
                    validate_background(image, "white")
            except Exception as e:
                quarantine(base, "white", image_id, str(e))

                if black_file.exists():
                    quarantine(
                        base, "black", image_id,
                        "The corresponding white-background image is invalid"
                    )

                state["attempts"]["white"][image_id] = max(
                    1,
                    state["attempts"]["white"].get(image_id, 0)
                )
                changed = True

        # Validate the black background.
        if black_file.exists():
            try:
                with Image.open(black_file) as image:
                    validate_background(image, "black")

                if white_file.exists():
                    with Image.open(white_file) as a:
                        with Image.open(black_file) as b:
                            if a.size != b.size:
                                raise ImageSizeMismatch(
                                    "White and black image dimensions do not match"
                                )

            except Exception as e:
                quarantine(base, "black", image_id, str(e))

                state["attempts"]["black"][image_id] = max(
                    1,
                    state["attempts"]["black"].get(image_id, 0)
                )
                changed = True

        for stage in ("white", "black"):
            if image_path(base, stage, image_id).exists():
                old = state["attempts"][stage].get(image_id, 0)

                if old == 0:
                    state["attempts"][stage][image_id] = 1
                    changed = True

    if changed:
        save_state(state, work)


# ==================== Alpha reconstruction ====================

def compose_transparent(white_file, black_file, out):
    with Image.open(white_file) as image:
        validate_background(image, "white")
        white = np.asarray(
            image.convert("RGB"), dtype=np.float32
        )

    with Image.open(black_file) as image:
        validate_background(image, "black")
        black = np.asarray(
            image.convert("RGB"), dtype=np.float32
        )

    if white.shape != black.shape:
        raise ImageSizeMismatch("White and black image dimensions do not match")

    difference = (white - black) / 255.0

    transmittance = np.clip(
        np.median(difference, axis=2), 0.0, 1.0
    )

    alpha = 1.0 - transmittance
    alpha[alpha < 0.01] = 0.0

    safe_alpha = np.maximum(
        alpha[..., None], 0.01
    )

    foreground = (
        white + black
        - (1.0 - alpha[..., None]) * 255.0
    ) / (2.0 * safe_alpha)

    foreground = np.clip(foreground, 0, 255)
    foreground[alpha == 0] = 0

    rgba = np.dstack([
        np.rint(foreground).astype(np.uint8),
        np.rint(alpha * 255).astype(np.uint8),
    ])

    temp = out.with_name(out.stem + ".tmp.png")

    Image.fromarray(rgba).save(
        temp, format="PNG"
    )

    os.replace(temp, out)


def compose_ready(records, base, state, work):
    for image_id, name, prompt in records:
        result = image_path(
            base, "transparent", image_id
        )

        if reuse_transparent(base, image_id):
            continue

        white = image_path(base, "white", image_id)
        black = image_path(base, "black", image_id)

        if not white.exists() or not black.exists():
            continue

        try:
            compose_transparent(white, black, result)
            print(f"  Transparent PNG saved: {image_id}.png")

        except Exception as e:
            print(f"  Composition failed for {image_id}: {e}")

            # Regenerate the black image first when dimensions do not match.
            if isinstance(e, ImageSizeMismatch):
                quarantine(base, "black", image_id, str(e))

            # Preserve images for inspection when composition fails otherwise.


# ==================== Batch construction ====================

def select_next_batch(
    records, base, state, batch_size, max_attempts
):
    white_pending = []
    black_pending = []

    for record in records:
        image_id = record[0]

        if transparent_output_error(image_path(
            base, "transparent", image_id
        )) is None:
            continue

        has_white = image_path(
            base, "white", image_id
        ).exists()

        has_black = image_path(
            base, "black", image_id
        ).exists()

        if (
            has_white
            and not has_black
            and state["attempts"]["black"].get(
                image_id, 0
            ) < max_attempts
        ):
            black_pending.append(record)

        elif (
            not has_white
            and state["attempts"]["white"].get(
                image_id, 0
            ) < max_attempts
        ):
            white_pending.append(record)

    # Finish images that already have a white-background source first.
    if black_pending:
        return "black", black_pending[:batch_size]

    if white_pending:
        return "white", white_pending[:batch_size]

    return None, []


def create_request_file(records, base, stage, size, token, black_prompt=BLACK_PROMPT):
    path = base / "_work" / (
        f"requests_{stage}_{token}.jsonl"
    )

    with path.open("w", encoding="utf-8") as f:
        for image_id, name, prompt in records:
            if stage == "white":
                parts = [
                    {"text": white_prompt(prompt)}
                ]
            else:
                white_file = image_path(
                    base, "white", image_id
                )

                image_data = base64.b64encode(
                    white_file.read_bytes()
                ).decode("ascii")

                parts = [
                    {"text": black_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_data,
                        }
                    },
                ]

            item = {
                "key": image_id,
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": parts,
                    }],
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                        "imageConfig": {
                            "aspectRatio": "1:1",
                            "imageSize": size,
                        },
                    },
                },
            }

            f.write(
                json.dumps(item, ensure_ascii=False)
                + "\n"
            )

    if path.stat().st_size > 2 * 1024**3:
        raise ValueError(
            "Batch input exceeds the official 2 GB limit"
        )

    return path


def prepare_batch(records, base, state, work, stage, size, black_prompt=BLACK_PROMPT):
    token = uuid.uuid4().hex[:16]

    request_file = create_request_file(
        records, base, stage, size, token, black_prompt
    )

    current = {
        "token": token,
        "stage": stage,
        "keys": [r[0] for r in records],
        "jsonl": str(request_file),
        "display_name": f"transparent-{stage}-{token}",
        "status": "prepared",
        "upload_name": None,
        "job_name": None,
        "counted": False,
    }

    state["current"] = current
    save_state(state, work)

    print(
        f"\nPreparing {stage} batch: "
        f"{len(records)} images, IDs "
        f"{records[0][0]} ~ {records[-1][0]}"
    )


# ==================== Safe batch submission ====================

def recover_uncertain_job(client, state, work, allow_retry):
    current = state["current"]
    display_name = current.get("display_name")

    if not display_name:
        raise StopRun("Cannot recover the job: display_name is missing")

    print("Checking whether the batch already exists on the server...")

    try:
        matches = [
            job for job in client.batches.list()
            if job.display_name == display_name
        ]
    except Exception as e:
        raise StopRun(
            f"Could not list server-side jobs: {e}. "
            "Paused to avoid duplicate charges."
        )

    if len(matches) > 1:
        raise StopRun(
            "Multiple batches with the same display name were found. "
            "Inspect them manually to avoid duplicate generation."
        )

    if len(matches) == 1:
        current["job_name"] = matches[0].name
        current["status"] = "submitted"
        save_state(state, work)

        print(f"Recovered job: {matches[0].name}")
        return

    if not allow_retry:
        raise StopRun(
            "Submission is uncertain and the job is not currently listed. "
            "Automatic resubmission is disabled to avoid duplicate charges. "
            "Check the batch list in AI Studio first. "
            "Only after confirming that no job was created, use"
            " --confirm-not-created."
        )

    print("Manual confirmation received: the job was not created. Resubmission is allowed.")

    current["status"] = "uploaded"
    save_state(state, work)


def ensure_submitted(client, state, work, allow_retry):
    current = state["current"]

    if current["status"] in ("creating", "uncertain"):
        recover_uncertain_job(
            client, state, work, allow_retry
        )

    if current["status"] == "prepared":
        print("Uploading batch JSONL...")

        try:
            uploaded = client.files.upload(
                file=current["jsonl"],
                config=types.UploadFileConfig(
                    display_name=current["display_name"],
                    mime_type="jsonl",
                ),
            )

        except Exception as e:
            if is_quota_error(e):
                raise StopRun(
                    f"Upload hit a 429 / quota limit: {e}"
                    "\nState saved. Rerun the same command to resume."
                )
            raise StopRun(
                f"File upload failed: {e}"
                "\nNo generation job was submitted; retrying is safe."
            )

        current["upload_name"] = uploaded.name
        current["status"] = "uploaded"
        save_state(state, work)

    if current["status"] == "uploaded":
        # Persist the creating state before calling the API.
        # A timed-out creation request must not be blindly resubmitted.
        current["status"] = "creating"
        save_state(state, work)

        print("Submitting the Gemini batch...")

        try:
            job = client.batches.create(
                model=state["model"],
                src=current["upload_name"],
                config={
                    "display_name": current["display_name"]
                },
            )

        except Exception as e:
            if is_quota_error(e):
                # The server explicitly rejected the request due to quota.
                current["status"] = "uploaded"
                save_state(state, work)

                raise StopRun(
                    f"Hit a 429 / quota limit: {e}"
                    "\nNo further batches were submitted."
                    "\nRerun the same command to resume submission."
                )

            # Other errors may occur after the job has already been created.
            current["status"] = "uncertain"
            save_state(state, work)

            raise StopRun(
                f"Batch creation returned an error: {e}"
                "\nIt is not yet clear whether the job was created."
                "\nThe next run will check the server-side job list first."
            )

        current["job_name"] = job.name
        current["status"] = "submitted"
        save_state(state, work)

        print(f"Batch created: {job.name}")

    if current["status"] != "submitted":
        raise StopRun("Unexpected current batch state")

    mark_attempts(state, current, work)


# ==================== Job polling ====================

def wait_for_batch(client, state):
    current = state["current"]
    job_name = current["job_name"]

    if not job_name:
        raise StopRun("The job has no job_name")

    while True:
        try:
            job = client.batches.get(name=job_name)

        except Exception as e:
            raise StopRun(
                f"Batch polling failed: {e}"
                "\nThe job ID is saved; the next run will resume polling."
            )

        status = getattr(
            job.state, "name", str(job.state)
        )

        print(
            f"  [{current['stage']}] "
            f"{job_name}: {status}",
            flush=True,
        )

        if status in TERMINAL_STATES:
            return job, status

        time.sleep(POLL_SECONDS)


# ==================== Result downloads ====================

def extract_image(response):
    for candidate in response.get("candidates", []):
        content = candidate.get("content") or {}

        for part in content.get("parts", []):
            if part.get("thought"):
                continue

            inline = (
                part.get("inlineData")
                or part.get("inline_data")
            )

            if inline and inline.get("data"):
                raw = base64.b64decode(
                    inline["data"]
                )

                with Image.open(io.BytesIO(raw)) as im:
                    return im.copy()

    return None


def download_result_file(file_name, result_file):
    import requests

    url = (
        "https://generativelanguage.googleapis.com"
        f"/download/v1beta/{file_name}:download?alt=media"
    )

    partial = Path(str(result_file) + ".part")
    mb = 1024 * 1024

    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0

        headers = {
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Accept-Encoding": "identity",
        }

        if offset:
            headers["Range"] = f"bytes={offset}-"

        print(
            f"Download attempt {attempt}/8; "
            f"{offset / mb:.1f} MB already downloaded",
            flush=True,
        )

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:

                if response.status_code == 416 and offset:
                    match = re.fullmatch(
                        r"bytes \*/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if match and offset == int(match.group(1)):
                        os.replace(partial, result_file)
                        return

                response.raise_for_status()

                if offset and response.status_code == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )

                    if not match or int(match.group(1)) != offset:
                        raise RuntimeError(
                            "The server returned an unexpected download range"
                        )

                    total = int(match.group(3))
                    mode = "ab"

                elif response.status_code == 200:
                    if offset:
                        print(
                            "The server does not support resuming this download; "
                            "restarting the full download."
                        )

                    offset = 0
                    mode = "wb"
                    total = int(
                        response.headers.get("Content-Length") or 0
                    )

                else:
                    raise RuntimeError(
                        f"Unexpected HTTP status: {response.status_code}"
                    )

                downloaded = offset
                last_report = offset

                if total:
                    print(
                        f"File size: {total / mb:.1f} MB",
                        flush=True,
                    )

                with partial.open(mode) as f:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not chunk:
                            continue

                        f.write(chunk)
                        downloaded += len(chunk)

                        if downloaded - last_report >= 5 * mb:
                            if total:
                                print(
                                    f"Download progress: "
                                    f"{downloaded / mb:.1f} / "
                                    f"{total / mb:.1f} MB",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"Downloaded: {downloaded / mb:.1f} MB",
                                    flush=True,
                                )

                            last_report = downloaded

                    f.flush()
                    os.fsync(f.fileno())

                if total and downloaded != total:
                    raise IOError(
                        f"Incomplete download: {downloaded}/{total}"
                    )

            os.replace(partial, result_file)
            print("Result file downloaded.", flush=True)
            return

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as e:
            print(f"Download interrupted: {e}", flush=True)

            if attempt < 8:
                time.sleep(min(attempt * 5, 30))

    raise StopRun(
        "Download failed after 8 attempts. "
        "The job ID and partial download have been preserved."
    )


def jsonl_lines(path):
    with path.open("r", encoding="utf-8") as f:
        yield from f

def harvest_results(client, job, base, state, work):
    current = state["current"]
    stage = current["stage"]
    token = current["token"]

    result_file = work / f"result_{token}.jsonl"

    # Reuse the result file if it has already been downloaded.
    if not result_file.exists():
        if not job.dest or not job.dest.file_name:
            raise StopRun("The batch has no result file")

        download_result_file(
            job.dest.file_name,
            result_file,
        )

    expected = set(current["keys"])
    quota_hit = False
    saved = 0
    failed = 0

    for line in jsonl_lines(result_file):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  Could not parse a result row: {e}")
            failed += 1
            continue

        image_id = item.get("key")

        if not image_id:
            metadata = item.get("metadata") or {}
            image_id = metadata.get("key")

        if image_id not in expected:
            print(f"  Ignoring an unmatched result: {image_id}")
            continue

        destination = image_path(
            base, stage, image_id
        )

        if destination.exists():
            continue

        error = item.get("error")

        if error:
            error_text = json.dumps(
                error, ensure_ascii=False
            )

            state["errors"][stage][image_id] = error_text
            failed += 1

            if is_quota_error(error_text):
                quota_hit = True

            print(f"  Generation failed for {image_id}: {error_text}")
            continue

        try:
            image = extract_image(
                item.get("response") or {}
            )

            if image is None:
                raise ValueError("No image was returned")

            validate_background(image, stage)

            temp = destination.with_name(
                image_id + ".tmp.png"
            )

            image.save(temp, format="PNG")
            os.replace(temp, destination)

            state["errors"][stage].pop(
                image_id, None
            )

            saved += 1

        except Exception as e:
            state["errors"][stage][image_id] = str(e)
            failed += 1
            print(f"  Invalid image for {image_id}: {e}")

    save_state(state, work)

    print(
        f"  Batch results: {saved} images saved; "
        f"{failed} failed or invalid images"
    )

    return quota_hit


def process_current(
    client, base, state, work, allow_retry
):
    current = state["current"]

    if current is None:
        return

    print(
        f"\nResuming/processing {current['stage']} "
        f"batch ({len(current['keys'])} images)"
    )

    ensure_submitted(
        client, state, work, allow_retry
    )

    job, status = wait_for_batch(client, state)

    quota_hit = False

    if job.dest and job.dest.file_name:
        quota_hit = harvest_results(
            client, job, base, state, work
        )

    elif status == "JOB_STATE_SUCCEEDED":
        raise StopRun(
            "The batch succeeded but returned no result file. "
            "The job state has been preserved for inspection."
        )

    # Clear the current job only after processing all available results.
    state["current"] = None
    save_state(state, work)

    if current["stage"] == "black":
        completed_batch = [(image_id, "", "") for image_id in current["keys"]]
        compose_ready(completed_batch, base, state, work)
        save_review(current["keys"], base, name=f"batch-{current['token']}")

    if status != "JOB_STATE_SUCCEEDED":
        raise StopRun(
            f"Batch finished with state {status}."
            f"\nError details: {job.error}"
            "\nAvailable results were saved. "
            "The next run will process only unfinished images."
        )

    if quota_hit:
        raise StopRun(
            "Some requests in the batch hit a quota limit. "
            "Successful images were saved and processing has paused. "
            "The next run will process the remaining IDs."
        )


# ==================== Summary report ====================

def save_review(image_ids, base, **kwargs):
    """Keep review rendering failures separate from paid generation state."""
    try:
        return write_review(image_ids, base, **kwargs)
    except (OSError, ValueError) as exc:
        print(f"Could not write review sheets: {exc}. Use --review-only to try again.")
        return []

def write_report(records, base, state, max_attempts):
    report = base / "failures.tsv"

    completed = 0
    remaining = []

    for image_id, name, prompt in records:
        final_file = image_path(
            base, "transparent", image_id
        )
        final_error = transparent_output_error(final_file)
        if final_error is None:
            completed += 1
            continue

        has_white = image_path(
            base, "white", image_id
        ).exists()

        has_black = image_path(
            base, "black", image_id
        ).exists()

        if not has_white:
            stage = "white"
        elif not has_black:
            stage = "black"
        else:
            stage = "compose"

        attempts = state["attempts"].get(
            stage, {}
        ).get(image_id, 0)

        if (
            stage in ("white", "black")
            and attempts >= max_attempts
        ):
            status = "Attempt limit reached"
        else:
            status = "Pending"

        error = state["errors"].get(
            stage, {}
        ).get(image_id, "")
        if final_file.exists() and final_error:
            error = f"Invalid transparent output: {final_error}" + (
                f"; {error}" if error else ""
            )

        remaining.append([
            image_id,
            name,
            stage,
            attempts,
            status,
            error,
        ])

    with report.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "ID", "Name", "Stage",
            "Attempts", "Status", "Error"
        ])

        writer.writerows(remaining)

    print("\n========== Summary ==========")
    print(f"Total: {len(records)}")
    print(f"Completed: {completed}")
    print(f"Remaining: {len(remaining)}")
    print(f"Final images: {base / 'transparent'}")
    print(f"Job state: {base / '_work' / 'state.json'}")
    print(f"Failure report: {report}")


# ==================== Entry point ====================

def main(argv=None, *, black_prompt=BLACK_PROMPT):
    parser = argparse.ArgumentParser(
        description="Generate white/black image pairs with Gemini Batch and reconstruct transparent PNGs."
    )

    parser.add_argument("tsv", help="Three-column TSV: ID, name, prompt")
    parser.add_argument("--out", default="output", help="Output directory (default: output)")
    parser.add_argument("--limit", type=int, help="Process only the first N records")
    parser.add_argument("--size", default="1K",
                        choices=["1K", "2K", "4K"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--review-only", action="store_true",
        help="Rebuild checkerboard review sheets from existing output without API calls"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS
    )

    parser.add_argument(
        "--confirm-not-created",
        action="store_true",
        help="Use only after manually confirming the job was not created on the server"
    )

    args = parser.parse_args(argv)

    if not args.review_only and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set"
        )

    if not 1 <= args.batch_size <= 100:
        raise ValueError("--batch-size must be between 1 and 100")

    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    records = load_tsv(args.tsv, args.limit)

    base = Path(args.out).resolve()

    for folder in (
        "white", "black", "transparent", "_work"
    ):
        (base / folder).mkdir(
            parents=True, exist_ok=True
        )

    with output_lock(base):
        if args.review_only:
            write_review([row[0] for row in records], base)
            return
        return run_batch(records, base, args, black_prompt)


def run_batch(records, base, args, black_prompt):
    """Run with the output lock held by the caller."""
    work = base / "_work"
    state = initialize_state(
        records, work, args.model, args.size
    )

    print(f"TSV records: {len(records)}")
    print(f"Model: {args.model}")
    print(f"Resolution: {args.size}")
    print(f"Images per batch: {args.batch_size}")
    print(f"Maximum attempts per stage: {args.max_attempts}")
    print(f"Directory: {base}")

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    try:
        while True:
            # Recover previously submitted jobs before starting new ones.
            if state.get("current"):
                process_current(
                    client,
                    base,
                    state,
                    work,
                    args.confirm_not_created,
                )
                continue

            # Validate existing files and quarantine invalid images.
            check_existing(
                records, base, state, work
            )

            # Compose existing white/black pairs without calling the API.
            compose_ready(
                records, base, state, work
            )

            stage, selected = select_next_batch(
                records,
                base,
                state,
                args.batch_size,
                args.max_attempts,
            )

            if not selected:
                print("\nNo more tasks are eligible for automatic submission.")
                break

            prepare_batch(
                selected,
                base,
                state,
                work,
                stage,
                args.size,
                black_prompt,
            )

            process_current(
                client,
                base,
                state,
                work,
                args.confirm_not_created,
            )

    except StopRun as e:
        print("\n========== Safely paused ==========")
        print(str(e))
        print(
            "Run the same command again to check the state and resume."
        )

    except KeyboardInterrupt:
        print("\nInterrupted. The state file has been preserved.")

    finally:
        try:
            write_report(records, base, state, args.max_attempts)
            save_review([row[0] for row in records], base)
        finally:
            client.close()


if __name__ == "__main__":
    main()
