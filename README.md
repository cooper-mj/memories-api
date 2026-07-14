# Render Backend – Wedding Memories

FastAPI service that generates presigned R2 upload URLs and runs face recognition on uploaded photos.

## Deploy to Render (free tier)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your repo and set the **Root Directory** to `render_app`.
4. Render auto-detects the `Dockerfile`. Build takes ~8 minutes on first deploy (dlib compilation).
5. Set these **Environment Variables** in Render's dashboard:

| Variable | Example value |
|---|---|
| `R2_ACCOUNT_ID` | `abc123def456` |
| `R2_ACCESS_KEY` | your R2 token key |
| `R2_SECRET_KEY` | your R2 token secret |
| `R2_BUCKET` | `wedding-memories` |
| `R2_PUBLIC_URL` | `https://pub-XXXX.r2.dev` |
| `ALLOWED_ORIGINS` | `https://cooper-mj.github.io,http://localhost:3000` |

6. Copy the service URL (e.g. `https://wedding-memories.onrender.com`) and set it as
   `NEXT_PUBLIC_RENDER_API_URL` in your Next.js build environment (or GitHub Actions secret).

> **Free tier note:** The service sleeps after 15 minutes of inactivity. The first upload of the day
> may take ~30 seconds to wake up. After that it's fast.

## R2 Bucket setup

1. Create an R2 bucket in Cloudflare dashboard.
2. Enable **Public access** on the bucket (or use a custom domain).
3. Set the public bucket URL as `R2_PUBLIC_URL`.
4. Add a **CORS policy** so the browser can PUT directly to R2:

```json
[
  {
    "AllowedOrigins": ["https://cooper-mj.github.io", "http://localhost:3000"],
    "AllowedMethods": ["PUT", "GET"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

5. Set `NEXT_PUBLIC_MANIFEST_URL` in your Next.js build to `https://pub-XXXX.r2.dev/manifest.json`.

## Initial face annotation

Run `scripts/annotate_faces.py` locally before deploying to build the face database:

```bash
cd ..
pip install face_recognition boto3 Pillow matplotlib
python scripts/annotate_faces.py \
  --photo path/to/mass_photo.jpg \
  --account-id YOUR_R2_ACCOUNT_ID \
  --access-key YOUR_R2_ACCESS_KEY \
  --secret-key YOUR_R2_SECRET_KEY \
  --bucket YOUR_BUCKET_NAME
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/people` | List of all tagged names |
| `GET` | `/presigned-upload?filename=&content_type=` | Get a presigned PUT URL |
| `POST` | `/process-upload` | Run face recognition on an uploaded photo |
| `POST` | `/reindex` | Re-tag all photos (run after adding new known faces) |
