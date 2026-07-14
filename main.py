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
from fastapi import FastAPI, HTTPException
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


def _identify_faces(encodings: list[np.ndarray]) -> list[str]:
    """Return names for each encoding (empty string if unrecognised)."""
    names: list[str] = []
    for enc in encodings:
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
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
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


@app.post("/process-upload")
def process_upload(body: ProcessUploadRequest):
    # Download from R2
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=body.key)
        img_bytes = obj["Body"].read()
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Object not found: {exc}")

    # Get dimensions from PIL (use client-supplied dims as fallback)
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        width, height = pil_img.size
    except Exception:
        width, height = body.width, body.height

    # Run face recognition
    try:
        img_array = face_recognition.load_image_file(io.BytesIO(img_bytes))
        encodings = face_recognition.face_encodings(img_array)
        identified = _identify_faces(encodings)
        people_in_photo = sorted({n for n in identified if n})
    except Exception as exc:
        print(f"[warn] face recognition failed for {body.key}: {exc}")
        people_in_photo = []

    # Update manifest
    manifest = _load_manifest()
    photo_url = f"{R2_PUBLIC_URL}/{body.key}"

    existing_urls = {p["url"] for p in manifest["photos"]}
    if photo_url not in existing_urls:
        manifest["photos"].append({
            "url": photo_url,
            "width": width,
            "height": height,
            "people": people_in_photo,
            "uploader": body.uploader,
        })
        all_people = set(manifest.get("people", [])) | set(people_in_photo)
        manifest["people"] = sorted(all_people)
        _save_manifest(manifest)

    return {"people_detected": people_in_photo, "photo_url": photo_url}


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
