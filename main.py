"""
Wedding Memories – Render backend
Handles presigned upload URLs and face-recognition tagging.
"""

from __future__ import annotations

import gc
import io
import json
import os
import threading
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

FACE_MATCH_THRESHOLD = 0.62  # dlib default is 0.6; slightly liberal for group photos
FACE_VOTE_TOP_K = 5          # use the K closest encodings per person when voting
THUMBNAIL_MAX_DIM = 900      # longest edge of display thumbnails

# Try these resolutions in order; below 600px we switch to patch mode instead.
_RECOGNITION_DIMS = [800, 600]
_PATCH_DIM = 500       # each patch is at most this wide/tall
_PATCH_OVERLAP = 0.15  # fraction of overlap between adjacent patches

# Only one recognition job runs at a time to stay within 512 MB RAM.
_recognition_lock = threading.Semaphore(1)


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
    """Identify each encoding using weighted voting across all known samples.

    For each candidate person we take their FACE_VOTE_TOP_K closest encodings,
    keep only those within FACE_MATCH_THRESHOLD, and sum (1 - distance) as a
    confidence score.  This exploits having many training samples per person and
    is more robust than a single nearest-neighbour lookup.
    """
    names: list[str] = []
    total = len(encodings)
    prefix = f"[{label}] " if label else ""

    for i, enc in enumerate(encodings, start=1):
        best_name = ""
        best_score = 0.0

        for name, known_encs in known_faces.items():
            if not known_encs:
                continue
            dists = face_recognition.face_distance(known_encs, enc)
            # Top-K closest, then filter by threshold
            top_k = np.sort(dists)[:FACE_VOTE_TOP_K]
            within = top_k[top_k < FACE_MATCH_THRESHOLD]
            if len(within) == 0:
                continue
            score = float(np.sum(1.0 - within))  # higher → more/closer matches
            if score > best_score:
                best_score = score
                best_name = name

        if best_name:
            print(f"{prefix}face {i}/{total} → {best_name} (score {best_score:.3f})")
        else:
            # Log the single closest person even when unmatched, to help tune threshold
            closest_name, closest_dist = "", 1.0
            for name, known_encs in known_faces.items():
                if not known_encs:
                    continue
                d = float(np.min(face_recognition.face_distance(known_encs, enc)))
                if d < closest_dist:
                    closest_dist = d
                    closest_name = name
            print(f"{prefix}face {i}/{total} → no match "
                  f"(closest: {closest_name or 'none'} @ {closest_dist:.3f})")
        names.append(best_name)

    return names


def _detect_faces(img_array: "np.ndarray", slug: str) -> list:
    """Detect face locations, trying upsample=2 first (finds smaller faces).
    Falls back to upsample=1 on OOM."""
    try:
        locs = face_recognition.face_locations(img_array, number_of_times_to_upsample=2, model="hog")
        print(f"[{slug}] detected {len(locs)} face(s) (upsample=2)")
        return locs
    except MemoryError:
        _oom_alert(slug, "OOM with upsample=2 during detection — falling back to upsample=1")
        gc.collect()
        locs = face_recognition.face_locations(img_array, number_of_times_to_upsample=1, model="hog")
        print(f"[{slug}] detected {len(locs)} face(s) (upsample=1)")
        return locs


def _oom_alert(slug: str, detail: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  ⚠️  OOM ERROR — {slug}")
    print(f"  {detail}")
    print(f"{bar}\n")


def _load_resized(img_bytes: bytes, max_dim: int) -> "np.ndarray":
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(pil.size) > max_dim:
        pil.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    del pil
    gc.collect()
    return face_recognition.load_image_file(buf)


def _encode_one_by_one(img_array: "np.ndarray", locations: list, slug: str) -> list["np.ndarray"]:
    """Encode faces one at a time to cap peak memory per face."""
    encodings: list[np.ndarray] = []
    for i, loc in enumerate(locations, start=1):
        enc = face_recognition.face_encodings(img_array, [loc])
        if enc:
            encodings.append(enc[0])
        print(f"[{slug}] encoded face {i}/{len(locations)}")
        gc.collect()
    return encodings


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union for two (top, right, bottom, left) boxes."""
    t = max(a[0], b[0]); r = min(a[1], b[1])
    bo = min(a[2], b[2]); l = max(a[3], b[3])
    if bo <= t or r <= l:
        return 0.0
    inter = (bo - t) * (r - l)
    area_a = (a[2] - a[0]) * (a[1] - a[3])
    area_b = (b[2] - b[0]) * (b[1] - b[3])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _try_at_dim(img_bytes: bytes, max_dim: int, slug: str) -> list["np.ndarray"]:
    """Attempt full-image recognition at max_dim. May raise MemoryError."""
    gc.collect()
    img_array = _load_resized(img_bytes, max_dim)
    locations = _detect_faces(img_array, slug)
    encodings = _encode_one_by_one(img_array, locations, slug)
    del img_array
    gc.collect()
    return encodings


def _recognize_patches(img_bytes: bytes, slug: str) -> list[str]:
    """Patch-based fallback: tile the image and merge results across patches."""
    print(f"[{slug}] switching to patch mode (patch_dim={_PATCH_DIM}px)")
    gc.collect()

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Keep overall image at 2× patch size so patches are meaningful
    max_overall = _PATCH_DIM * 2
    if max(pil.size) > max_overall:
        pil.thumbnail((max_overall, max_overall), Image.LANCZOS)
    W, H = pil.size

    # Build patch grid with overlap
    stride = int(_PATCH_DIM * (1 - _PATCH_OVERLAP))
    xs = list(range(0, W - 1, stride)) if W > _PATCH_DIM else [0]
    ys = list(range(0, H - 1, stride)) if H > _PATCH_DIM else [0]

    patches = [(x, y, min(x + _PATCH_DIM, W), min(y + _PATCH_DIM, H)) for y in ys for x in xs]
    print(f"[{slug}] {len(patches)} patches over {W}×{H} image")

    global_locs: list[tuple] = []
    all_encodings: list[np.ndarray] = []

    for pi, (x1, y1, x2, y2) in enumerate(patches, start=1):
        try:
            gc.collect()
            patch_pil = pil.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            patch_pil.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            del patch_pil

            patch_arr = face_recognition.load_image_file(buf)
            locs = _detect_faces(patch_arr, f"{slug}/p{pi}")

            for loc in locs:
                top, right, bottom, left = loc
                g_loc = (top + y1, right + x1, bottom + y1, left + x1)
                if any(_iou(g_loc, gl) > 0.3 for gl in global_locs):
                    continue  # duplicate across patch boundary
                enc = face_recognition.face_encodings(patch_arr, [loc])
                if enc:
                    global_locs.append(g_loc)
                    all_encodings.append(enc[0])
                gc.collect()

            del patch_arr
            gc.collect()

        except MemoryError:
            _oom_alert(slug, f"OOM inside patch {pi}/{len(patches)} — patch skipped")
            gc.collect()

    del pil
    gc.collect()
    print(f"[{slug}] patch mode done: {len(all_encodings)} unique face(s)")
    return _identify_faces(all_encodings, label=slug)


def _recognize_faces_safe(img_bytes: bytes, slug: str) -> list[str]:
    """Try full-image recognition at each resolution; fall back to patch mode on OOM."""
    for dim in _RECOGNITION_DIMS:
        try:
            encodings = _try_at_dim(img_bytes, dim, slug)
            return _identify_faces(encodings, label=slug)
        except MemoryError:
            _oom_alert(slug, f"OOM at {dim}px full-image — trying next resolution")
            gc.collect()
        except Exception as exc:
            print(f"[{slug}] recognition error at {dim}px: {exc}")
            return []

    # All full-image attempts failed — go to patch mode
    _oom_alert(slug, "OOM at all full-image resolutions — switching to patch mode")
    try:
        return _recognize_patches(img_bytes, slug)
    except MemoryError:
        _oom_alert(slug, "OOM even in patch mode — photo saved without face tags")
        gc.collect()
        return []
    except Exception as exc:
        print(f"[{slug}] patch mode error: {exc}")
        return []


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

    slug = body.key.split("/")[-1]
    print(f"[{slug}] waiting for recognition slot…")
    with _recognition_lock:
        identified = _recognize_faces_safe(img_bytes, slug)
        people_in_photo = sorted({n for n in identified if n})

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
            key = photo["url"].replace(f"{R2_PUBLIC_URL}/", "")
            slug = key.split("/")[-1]
            obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
            img_bytes = obj["Body"].read()
            identified = _recognize_faces_safe(img_bytes, slug)
            photo["people"] = sorted({n for n in identified if n})
            photo["indexed"] = True
            updated += 1
        except Exception as exc:
            print(f"[warn] reindex failed for {photo.get('url')}: {exc}")

    all_people: set[str] = set()
    for photo in manifest["photos"]:
        all_people.update(photo["people"])
    manifest["people"] = sorted(all_people)
    _save_manifest(manifest)

    return {"reindexed": updated, "total_people": len(manifest["people"])}
