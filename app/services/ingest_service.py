import asyncio
import hashlib
import math
import httpx
from typing import List, Dict
from simhash import Simhash
from sqlalchemy import select
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.database import AsyncSessionLocal, init_db
from app.models.knowledge import KnowledgeChunk


# 在文件顶部（imports 下方）创建一个全局复用的异步连接池
http_client = httpx.AsyncClient(timeout=10.0)

async def get_embedding(text: str) -> List[float]:
    """生成 1536 维向量（复用长连接池，消除重复 TLS 握手耗时）"""
    if not settings.EMBEDDING_API_KEY or settings.EMBEDDING_API_KEY == "none":
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[i % 32] / 255.0) - 0.5) for i in range(1536)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]
    
    url = f"{settings.EMBEDDING_BASE_URL}/embeddings/multimodal"
    headers = {
        "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "input": [{"type": "text", "text": text}]
    }
    # 直接复用全局 http_client，不再每次 with新建
    resp = await http_client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    vec = data["data"]["embedding"]
    if len(vec) < 1536:
        vec = vec + [0.0] * (1536 - len(vec))
    return vec[:1536]

# 初始化异步向量客户端
embed_client = AsyncOpenAI(
    api_key=settings.EMBEDDING_API_KEY,
    base_url=settings.EMBEDDING_BASE_URL
)

def compute_simhash(text: str) -> str:
    """计算文本的 64 位 SimHash 指纹"""
    return str(Simhash(text).value)

def is_duplicate(new_fp: str, existing_fps: List[str], threshold: int = 5) -> bool:
    """计算海明距离（Hamming Distance），距离 <= 5 判定为高度重复话术"""
    new_val = int(new_fp)
    for old_fp in existing_fps:
        old_val = int(old_fp)
        # 异或后统计 1 的个数，即为不同二进制位的数量
        distance = bin(new_val ^ old_val).count("1")
        if distance <= threshold:
            return True
    return False



async def get_embedding(text: str) -> List[float]:
    """生成 1536 维向量（支持火山方舟 Doubao-embedding-vision 与离线双模式）"""
    if not settings.EMBEDDING_API_KEY or settings.EMBEDDING_API_KEY == "none":
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[i % 32] / 255.0) - 0.5) for i in range(1536)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]
    
    url = f"{settings.EMBEDDING_BASE_URL}/embeddings/multimodal"
    headers = {
        "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "input": [{"type": "text", "text": text}]
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        vec = data["data"]["embedding"]
        # 自动对齐数据库 1536 维字段
        if len(vec) < 1536:
            vec = vec + [0.0] * (1536 - len(vec))
        return vec[:1536]

# 模拟跨境电商真实知识库语料（故意混入 2 条高度重复话术，用来验证指纹去重效果！）
MOCK_ECOMMERCE_DOCS: List[Dict] = [
    {
        "sku_id": "SKU-WT01",
        "category": "product",
        "language": "en",
        "content": "SKU-WT01 Pro Wireless Noise-Canceling Headphones feature 40-hour battery life, IPX5 water resistance, and Bluetooth 5.3 multipoint connection."
    },
    {
        "sku_id": "SKU-WT01",
        "category": "product",
        "language": "en",
        "content": "SKU-WT01 Pro Wireless Noise-Canceling Headphones feature 40-hour battery life, IPX5 waterproof rating, and Bluetooth 5.3 multipoint connection."  # 故意高度重复！
    },
    {
        "sku_id": "SKU-WT01",
        "category": "troubleshoot",
        "language": "en",
        "content": "To reset SKU-WT01 headphones: hold the Power and Volume Up buttons simultaneously for 8 seconds until the LED flashes red and blue three times."
    },
    {
        "sku_id": "ALL",
        "category": "return",
        "language": "en",
        "content": "Global Return Policy: Buyers can request a free return or full refund within 30 days of delivery if the item is unworn and in original packaging. Damaged items get instant replacement."
    },
    {
        "sku_id": "ALL",
        "category": "return",
        "language": "en",
        "content": "Global Return Policy: Buyers can request a free return and full refund within 30 days of delivery if the item is unworn in its original packaging. Damaged items get instant replacement."  # 故意高度重复！
    },
    {
        "sku_id": "ALL",
        "category": "logistics",
        "language": "en",
        "content": "US & EU Express Shipping takes 5-8 business days via DHL/USPS. Customs duties are prepaid (DDP) for all orders over $50."
    },
    {
        "sku_id": "SKU-SW09",
        "category": "product",
        "language": "en",
        "content": "SKU-SW09 Smart Fitness Watch supports 5ATM swimming waterproof, 24/7 heart rate and SpO2 monitoring, compatible with both iOS 14+ and Android 10+."
    }
]

async def run_ingest():
    await init_db()
    async with AsyncSessionLocal() as session:
        # 查出数据库里已有的所有指纹
        result = await session.execute(select(KnowledgeChunk.simhash_fingerprint))
        existing_fps = [row[0] for row in result.all()]

        inserted_count = 0
        skipped_count = 0

        for doc in MOCK_ECOMMERCE_DOCS:
            fp = compute_simhash(doc["content"])
            # 1. 执行 SimHash 海明距离去重拦截
            if is_duplicate(fp, existing_fps, threshold=5):
                skipped_count += 1
                print(f"🚫 [指纹去重拦截] 发现重复知识切片，已跳过: {doc['content'][:45]}...")
                continue

            # 2. 生成向量并入库
            vec = await get_embedding(doc["content"])
            chunk = KnowledgeChunk(
                sku_id=doc["sku_id"],
                category=doc["category"],
                language=doc["language"],
                content=doc["content"],
                simhash_fingerprint=fp,
                embedding=vec
            )
            session.add(chunk)
            existing_fps.append(fp)
            inserted_count += 1
            print(f"✅ [成功入库] SKU={doc['sku_id']} | 分类={doc['category']}")

        await session.commit()
        print(f"\n🎉 知识库入库完成！共处理 {len(MOCK_ECOMMERCE_DOCS)} 条，成功入库 {inserted_count} 条，指纹去重拦截 {skipped_count} 条。")

if __name__ == "__main__":
    asyncio.run(run_ingest())