FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

# dlib-bin provides a pre-compiled dlib wheel — no cmake or C++ compilation needed.
# Install it first, then face_recognition with --no-deps so pip doesn't try to
# rebuild dlib from source to satisfy the dependency.
RUN pip install --no-cache-dir dlib-bin face_recognition_models
RUN pip install --no-cache-dir face_recognition --no-deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
