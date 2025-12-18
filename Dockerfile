# CLIP Ranker API - Dockerfile (CPU PoC)
FROM python:3.11-slim

WORKDIR /app

# システム依存（PIL用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 依存インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリコード
COPY app.py .

# uvicorn起動（ポート8000）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

