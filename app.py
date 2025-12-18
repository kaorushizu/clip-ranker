"""
CLIP Ranker API - PoC
画像類似度ランキングAPI（OpenCLIP ViT-B/32, CPU）
"""

import asyncio
import base64
import io
from typing import Optional

import httpx
import numpy as np
import open_clip
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="CLIP Ranker API", version="0.1.0")

# ─────────────────────────────────────────────────────────────
# Model loading (global, loaded once at startup)
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cpu"
BATCH_SIZE = 16

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
)
model.eval()

# ─────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────
class Candidate(BaseModel):
    auction_id: str
    image_url: str
    title: Optional[str] = None


class RankRequest(BaseModel):
    query_image_base64: Optional[str] = None  # base64エンコード画像（優先）
    query_image_url: Optional[str] = None      # 画像URL（base64がない場合に使用）
    candidates: list[Candidate]
    top_k: int = 50
    max_concurrency: int = 50
    image_resize: int = 768


class RankResultItem(BaseModel):
    auction_id: str
    score: float
    rank: int


class FailedItem(BaseModel):
    auction_id: str
    reason: str


class MetaInfo(BaseModel):
    model: str
    image_resize: int
    candidates: int
    succeeded: int
    failed: int


class RankResponse(BaseModel):
    results: list[RankResultItem]
    failed: list[FailedItem]
    meta: MetaInfo


# ─────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────
USER_AGENT = "CLIP-Ranker-PoC/0.1 (compatible; httpx)"


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """長辺を max_size にアスペクト維持でリサイズ"""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def load_image_from_bytes(data: bytes, max_size: int) -> Image.Image:
    """バイナリから画像を読み込み、リサイズ＋RGB変換"""
    img = Image.open(io.BytesIO(data))
    img = resize_image(img, max_size)
    return img.convert("RGB")


async def download_image(
    client: httpx.AsyncClient,
    url: str,
    max_size: int,
    semaphore: asyncio.Semaphore,
) -> Image.Image:
    """
    画像をダウンロードしてPIL Imageを返す。
    - follow_redirects=True
    - User-Agent ヘッダ付与
    - timeout 3秒、retry 1回
    """
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(2):  # 最大2回（初回 + retry 1回）
        try:
            async with semaphore:
                resp = await client.get(
                    url,
                    headers=headers,
                    timeout=3.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                return load_image_from_bytes(resp.content, max_size)
        except Exception as e:
            last_exc = e
            if attempt == 0:
                await asyncio.sleep(0.1)  # 少し待ってリトライ
    raise last_exc  # type: ignore


def get_embedding(images: list[Image.Image]) -> np.ndarray:
    """
    画像リストから埋め込みベクトルを取得（バッチ推論）。
    戻り値は (N, D) の L2正規化済み numpy 配列。
    """
    all_embeddings = []
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i : i + BATCH_SIZE]
        tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(tensors)
        # L2 正規化
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())
    return np.vstack(all_embeddings)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return "ok"


@app.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    # 1. クエリ画像を取得（base64 優先、なければ URL からダウンロード）
    if req.query_image_base64:
        try:
            query_bytes = base64.b64decode(req.query_image_base64)
            query_img = load_image_from_bytes(query_bytes, req.image_resize)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid query_image_base64: {e}")
    elif req.query_image_url:
        try:
            async with httpx.AsyncClient() as client:
                headers = {"User-Agent": USER_AGENT}
                resp = await client.get(
                    req.query_image_url,
                    headers=headers,
                    timeout=5.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                query_img = load_image_from_bytes(resp.content, req.image_resize)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download query_image_url: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Either query_image_base64 or query_image_url must be provided"
        )

    # 2. 候補画像を並列ダウンロード
    semaphore = asyncio.Semaphore(req.max_concurrency)
    succeeded: list[tuple[str, Image.Image]] = []
    failed: list[FailedItem] = []

    limits = httpx.Limits(
        max_connections=req.max_concurrency,
        max_keepalive_connections=req.max_concurrency,
    )
    async with httpx.AsyncClient(limits=limits) as client:

        async def fetch_one(cand: Candidate):
            try:
                img = await download_image(
                    client, cand.image_url, req.image_resize, semaphore
                )
                return (cand.auction_id, img, None)
            except Exception as e:
                return (cand.auction_id, None, str(e))

        tasks = [fetch_one(c) for c in req.candidates]
        results = await asyncio.gather(*tasks)

    for auction_id, img, error in results:
        if img is not None:
            succeeded.append((auction_id, img))
        else:
            failed.append(FailedItem(auction_id=auction_id, reason=error or "unknown"))

    # 3. 埋め込み計算（クエリ + 成功分候補をバッチ推論）
    if not succeeded:
        # 全部失敗した場合
        return RankResponse(
            results=[],
            failed=failed,
            meta=MetaInfo(
                model=f"openclip_{MODEL_NAME.lower().replace('-', '_')}",
                image_resize=req.image_resize,
                candidates=len(req.candidates),
                succeeded=0,
                failed=len(failed),
            ),
        )

    # クエリ埋め込み
    query_emb = get_embedding([query_img])  # (1, D)

    # 候補埋め込み
    cand_ids = [aid for aid, _ in succeeded]
    cand_imgs = [img for _, img in succeeded]
    cand_embs = get_embedding(cand_imgs)  # (N, D)

    # 4. コサイン類似度（正規化済みなので内積）
    scores = (cand_embs @ query_emb.T).flatten()  # (N,)

    # 5. スコア順にソートしてランク付与
    sorted_indices = np.argsort(-scores)  # 降順
    rank_results: list[RankResultItem] = []
    for rank_idx, idx in enumerate(sorted_indices):
        if rank_idx >= req.top_k:
            break
        rank_results.append(
            RankResultItem(
                auction_id=cand_ids[idx],
                score=round(float(scores[idx]), 4),
                rank=rank_idx + 1,
            )
        )

    return RankResponse(
        results=rank_results,
        failed=failed,
        meta=MetaInfo(
            model=f"openclip_{MODEL_NAME.lower().replace('-', '_')}",
            image_resize=req.image_resize,
            candidates=len(req.candidates),
            succeeded=len(succeeded),
            failed=len(failed),
        ),
    )

