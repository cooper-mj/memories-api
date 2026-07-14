"""
Wedding Memories – Render backend
Handles presigned upload URLs and face-recognition tagging.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import boto3
import face_recognition
import numpy as np
from botocore.config import Config
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ── R2 Config (set as Render environment variables) ───────────────────────────
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]
R2_PUBLIC_URL = os.environ["R2_PUBLIC_URL"].rstrip("/")  # e.g. https://pub-xxx.r2.dev
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://cooper-mj.github.io,http://localhost:3000",
).split(",")
# ─────────────────────────────────────────────────────────────────────────────

FACE_MATCH_THRESHOLD = 0.55  # lower = stricter (0.6 is dlib default)
THUMBNAIL_MAX_DIM = 900      # longest edge of display thumbnails


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


s3 = make_s3()

# In-memory cache – populated at startup and refreshed after each reindex.
known_faces: dict[str, list[np.ndarray]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_json(key: str, default: Any = None) -> Any:
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return default
    except Exception as exc:
        print(f"[warn] could not fetch {key}: {exc}")
        return default


def _put_json(key: str, data: Any) -> None:
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode(),
        ContentType="application/json",
    )


def _load_known_faces() -> None:
    global known_faces
    raw = _get_json("known_faces.json", {})
    known_faces = {
        name: [np.array(enc) for enc in encodings]
        for name, encodings in raw.items()
    }
    print(f"[startup] loaded {len(known_faces)} known identities")


def _save_known_faces() -> None:
    serialisable = {
        name: [enc.tolist() for enc in encs]
        for name, encs in known_faces.items()
    }
    _put_json("known_faces.json", serialisable)


def _load_manifest() -> dict:
    return _get_json("manifest.json", {"people": [], "photos": []})


def _save_manifest(manifest: dict) -> None:
    _put_json("manifest.json", manifest)


def _identify_faces(encodings: list[np.ndarray], label: str = "") -> list[str]:
    """Return names for each encoding, logging progress face-by-face."""
    names: list[str] = []
    total = len(encodings)
    prefix = f"[{label}] " if label else ""
    for i, enc in enumerate(encodings, start=1):
        best_name = ""
        best_dist = FACE_MATCH_THRESHOLD
        for name, known_encs in known_faces.items():
            if not known_encs:
                continue
            dists = face_recognition.face_distance(known_encs, enc)
            d = float(np.min(dists))
            if d < best_dist:
                best_dist = d
                best_name = name
        if best_name:
            print(f"{prefix}face {i}/{total} → {best_name} (dist {best_dist:.3f})")
        else:
            print(f"{prefix}face {i}/{total} → no match (closest dist {best_dist:.3f})")
        names.append(best_name)
    return names


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_known_faces()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "known_identities": len(known_faces)}


@app.get("/people")
def get_people():
    manifest = _load_manifest()
    return {"people": manifest.get("people", [])}


@app.get("/presigned-upload")
def presigned_upload(filename: str, content_type: str = "image/jpeg"):
    # Always store under photos/ with a UUID prefix to avoid collisions.
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
    key = f"photos/{uuid.uuid4().hex}.{ext}"
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": R2_BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=600,
    )
    return {"url": url, "key": key}


class ProcessUploadRequest(BaseModel):
    key: str
    uploader: str = "Anonymous"
    width: int = 0
    height: int = 0


MAX_RECOGNITION_DIM = 1000  # downscale before face recognition to stay within 512MB RAM


def _make_thumbnail(img_bytes: bytes) -> tuple[bytes, int, int]:
    """Return (jpeg_bytes, width, height) for the thumbnail."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(pil.size) > THUMBNAIL_MAX_DIM:
        pil.thumbnail((THUMBNAIL_MAX_DIM, THUMBNAIL_MAX_DIM), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), pil.width, pil.height


def _do_process_upload(body: ProcessUploadRequest) -> None:
    """Runs in background — download photo, create thumbnail, tag faces, update manifest."""
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=body.key)
        img_bytes = obj["Body"].read()
    except Exception as exc:
        print(f"[error] could not download {body.key}: {exc}")
        return

    # Get true dimensions from the original image
    try:
        pil_orig = Image.open(io.BytesIO(img_bytes))
        orig_width, orig_height = pil_orig.size
    except Exception:
        orig_width, orig_height = body.width, body.height

    # Generate and upload thumbnail
    thumb_url = None
    try:
        thumb_bytes, thumb_w, thumb_h = _make_thumbnail(img_bytes)
        thumb_key = "thumbnails/" + body.key.split("/", 1)[-1]
        s3.put_object(Bucket=R2_BUCKET, Key=thumb_key, Body=thumb_bytes, ContentType="image/jpeg")
        thumb_url = f"{R2_PUBLIC_URL}/{thumb_key}"
    except Exception as exc:
        print(f"[warn] thumbnail generation failed: {exc}")
        thumb_w, thumb_h = orig_width, orig_height

    # Add photo to manifest immediately so it shows in the gallery even if face
    # recognition crashes below.
    photo_url = f"{R2_PUBLIC_URL}/{body.key}"
    manifest = _load_manifest()
    existing_urls = {p["url"] for p in manifest["photos"]}
    if photo_url not in existing_urls:
        manifest["photos"].append({
            "url": photo_url,
            "thumbnail": thumb_url,
            "width": thumb_w,
            "height": thumb_h,
            "people": [],
            "indexed": False,
            "uploader": body.uploader,
        })
        _save_manifest(manifest)
        print(f"[manifest] added {body.key} (face tagging pending)")

    # Downscale before recognition to avoid OOM on large photos
    slug = body.key.split("/")[-1]
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if max(pil_img.size) > MAX_RECOGNITION_DIM:
            pil_img.thumbnail((MAX_RECOGNITION_DIM, MAX_RECOGNITION_DIM), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        print(f"[{slug}] detecting faces…")
        img_array = face_recognition.load_image_file(buf)
        encodings = face_recognition.face_encodings(img_array)
        print(f"[{slug}] found {len(encodings)} face(s), identifying…")
        identified = _identify_faces(encodings, label=slug)
        people_in_photo = sorted({n for n in identified if n})
    except Exception as exc:
        print(f"[warn] face recognition failed for {body.key}: {exc}")
        people_in_photo = []

    # Update manifest entry: mark indexed, add people
    manifest = _load_manifest()
    for photo in manifest["photos"]:
        if photo["url"] == photo_url:
            photo["people"] = people_in_photo
            photo["indexed"] = True
            break
    all_people = set(manifest.get("people", [])) | set(people_in_photo)
    manifest["people"] = sorted(all_people)
    _save_manifest(manifest)

    print(f"[done] {body.key} → {people_in_photo or '(no face matches)'}")


@app.post("/process-upload")
def process_upload(body: ProcessUploadRequest, background_tasks: BackgroundTasks):
    """Returns immediately; face recognition runs in the background."""
    background_tasks.add_task(_do_process_upload, body)
    return {"status": "queued", "key": body.key}


@app.post("/reindex")
def reindex():
    """Re-run face recognition on every photo in the manifest.
    Useful after updating known_faces.json with new annotations."""
    _load_known_faces()
    manifest = _load_manifest()
    updated = 0

    for photo in manifest["photos"]:
        try:
            # Extract key from URL
            key = photo["url"].replace(f"{R2_PUBLIC_URL}/", "")
            obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
            img_bytes = obj["Body"].read()
            img_array = face_recognition.load_image_file(io.BytesIO(img_bytes))
            encodings = face_recognition.face_encodings(img_array)
            identified = _identify_faces(encodings)
            photo["people"] = sorted({n for n in identified if n})
            updated += 1
        except Exception as exc:
            print(f"[warn] reindex failed for {photo.get('url')}: {exc}")

    all_people: set[str] = set()
    for photo in manifest["photos"]:
        all_people.update(photo["people"])
    manifest["people"] = sorted(all_people)
    _save_manifest(manifest)

    return {"reindexed": updated, "total_people": len(manifest["people"])}
