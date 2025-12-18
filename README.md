# CLIP Ranker API (PoC)

画像類似度ランキングAPI（OpenCLIP ViT-B/32, CPU版）

## 概要

- Expoアプリから「クエリ画像(base64)」と「候補画像URL一覧」を送ると、類似度順でランキングを返す
- FastAPI + OpenCLIP + Docker で動作
- VPS上で Caddy の reverse_proxy 経由で公開

---

## VPSへのデプロイ手順

### 1. ファイル配置

このリポジトリの内容を VPS の `/root/clip-ranker` に配置します。

```bash
# VPS上で
mkdir -p /root/clip-ranker
# scpやgit cloneなどでファイルを配置
```

配置後のディレクトリ構成:

```
/root/clip-ranker/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### 2. Caddyfile の変更

`/root/revathis-ai-agents/Caddyfile` を編集し、`clip.revathis-dev.com` のブロックを以下に変更:

```caddyfile
clip.revathis-dev.com {
  reverse_proxy clip-api:8000
}
```

変更後、Caddyを再起動:

```bash
docker restart revathis-ai-caddy
```

### 3. コンテナ起動

```bash
cd /root/clip-ranker
docker compose up -d --build
```

ログ確認:

```bash
docker logs clip-api --tail=50
```

---

## 動作確認

### 内部疎通（Caddyコンテナ → clip-api）

```bash
docker exec -it revathis-ai-caddy wget -qO- http://clip-api:8000/healthz
```

期待される出力: `ok`

### 外部疎通（HTTPS経由）

```bash
curl -s https://clip.revathis-dev.com/healthz
```

期待される出力: `"ok"`

---

## /rank API のテスト

### リクエスト形式

```json
{
  "query_image_base64": "<base64エンコードされた画像>",
  "candidates": [
    { "auction_id": "A1", "image_url": "https://example.com/img1.jpg", "title": "商品1" },
    { "auction_id": "A2", "image_url": "https://example.com/img2.jpg", "title": "商品2" }
  ],
  "top_k": 50,
  "max_concurrency": 10,
  "image_resize": 768
}
```

### curl テスト例

まず、テスト用の base64 画像を用意します:

```bash
# ローカルの画像をbase64に変換（例）
BASE64_IMG=$(base64 -w0 /path/to/your/test-image.jpg)
# macOSの場合: BASE64_IMG=$(base64 -i /path/to/your/test-image.jpg)
```

APIリクエスト:

```bash
curl -s -X POST https://clip.revathis-dev.com/rank \
  -H "Content-Type: application/json" \
  -d '{
    "query_image_base64": "'"${BASE64_IMG}"'",
    "candidates": [
      {"auction_id": "test1", "image_url": "https://picsum.photos/id/237/400/300", "title": "Dog"},
      {"auction_id": "test2", "image_url": "https://picsum.photos/id/1084/400/300", "title": "Cat"},
      {"auction_id": "test3", "image_url": "https://picsum.photos/id/1074/400/300", "title": "Mountain"}
    ],
    "top_k": 10,
    "max_concurrency": 10,
    "image_resize": 768
  }' | jq .
```

### レスポンス例

```json
{
  "results": [
    { "auction_id": "test1", "score": 0.8732, "rank": 1 },
    { "auction_id": "test2", "score": 0.8124, "rank": 2 },
    { "auction_id": "test3", "score": 0.6543, "rank": 3 }
  ],
  "failed": [],
  "meta": {
    "model": "openclip_vit_b_32",
    "image_resize": 768,
    "candidates": 3,
    "succeeded": 3,
    "failed": 0
  }
}
```

---

## トラブルシューティング

### コンテナが起動しない

```bash
docker logs clip-api
```

でエラーを確認。モデルのダウンロードに時間がかかる場合があります（初回起動時）。

### 外部からアクセスできない

1. Caddyfile の設定を確認
2. `docker restart revathis-ai-caddy` を実行
3. ネットワーク確認: `docker network inspect revathis-ai-agents_default`

### 画像ダウンロードが失敗する

`failed` 配列にエラー理由が入ります。よくある原因:
- 画像URLが無効
- リダイレクトが多すぎる
- サーバーがUser-Agentを拒否している

---

## 技術仕様

| 項目 | 値 |
|------|-----|
| モデル | OpenCLIP ViT-B-32 (openai pretrained) |
| 推論デバイス | CPU |
| バッチサイズ | 16 |
| 画像リサイズ | 長辺768px（アスペクト維持） |
| ダウンロードタイムアウト | 3秒 |
| リトライ | 1回 |

---

## 次フェーズ（予定）

- [ ] より大きいモデル（ViT-L/14, SigLIP等）への切り替え
- [ ] GPU対応
- [ ] タイトルテキストの併用（画像×テキストマルチモーダル）
- [ ] 閾値フィルタリング
- [ ] キャッシュ（埋め込みのメモ化）

