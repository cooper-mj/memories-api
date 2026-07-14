FROM python:3.11-slim

# dlib (required by face_recognition) needs cmake + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Build dlib wheel first (slow, ~5 min) then the rest
RUN pip install --no-cache-dir cmake dlib
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
