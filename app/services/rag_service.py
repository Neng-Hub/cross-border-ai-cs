import asyncio
import re
import time
import jieba
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from sqlalchemy import select, or_

from app.core.database import AsyncSessionLocal
from app.models.knowledge import KnowledgeChunk
from app.services.ingest_service import get_embedding


def extract_sku_from_query(query: str) -> Optional[str]:
    """用正则从用户提问中提取 SKU 编号（例如 SKU-WT01），用于前置元数据硬过滤"""
    match = re.search(r"SKU-[A-Z0-9]+", query.upper())
    return match.group(0) if match else None


def tokenize(text: str) -> List[str]:
    """中英文混合分词，供 BM25 关键词检索使用"""
    return [w.strip().lower() for w in jieba.cut(text) if w.strip()]


class HybridRAGService:
    def __init__(self):
        # 统计指标：方便面试展示或接口监控查看降级触发率
        self.total_queries = 0
        self.degraded_queries = 0

    async def search(
        self,
        query: str,
        top_k: int = 3,
        embed_timeout: float = 0.3,  # 核心参数：向量化超时阈值 300ms (0.3秒)
        simulate_slow_embed: bool = False  # 测试开关：人为制造延迟，验证超时降级
    ) -> Dict:
        self.total_queries += 1
        start_time = time.perf_counter()
        degraded = False
        degrade_reason = ""

        # 1. 前置元数据硬过滤：识别用户问题里有没有提到具体 SKU
        target_sku = extract_sku_from_query(query)

        async with AsyncSessionLocal() as session:
            # 构造 SQL 过滤条件：如果提到了 SKU-WT01，就只查该 SKU 和通用政策(ALL)
            stmt = select(KnowledgeChunk)
            if target_sku:
                stmt = stmt.where(
                    or_(
                        KnowledgeChunk.sku_id == target_sku,
                        KnowledgeChunk.sku_id == "ALL"
                    )
                )

            result = await session.execute(stmt)
            candidate_chunks: List[KnowledgeChunk] = list(result.scalars().all())

            if not candidate_chunks:
                return {"results": [], "degraded": False, "latency_ms": 0}

            # 2. 路数一：BM25 稀疏关键词检索（本地内存毫秒级计算，永不超时）
            corpus_tokens = [tokenize(c.content) for c in candidate_chunks]
            bm25 = BM25Okapi(corpus_tokens)
            query_tokens = tokenize(query)
            bm25_scores = bm25.get_scores(query_tokens)

            # 按 BM25 分数从高到低排序，记录排名名次
            bm25_ranked = sorted(
                zip(candidate_chunks, bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )
            bm25_rank_map = {chunk.id: idx + 1 for idx, (chunk, _) in enumerate(bm25_ranked)}

            # 3. 路数二：pgvector 向量检索（包裹 300ms 超时熔断保护！）
            vector_rank_map = {}
            try:
                async def _fetch_vector_ranks():
                    if simulate_slow_embed:
                        await asyncio.sleep(0.5)  # 故意睡 500ms，模拟网络抖动超时
                    q_vec = await get_embedding(query)

                    # 使用 pgvector 的余弦距离 (<=>) 在数据库层面做向量排序
                    vec_stmt = select(KnowledgeChunk).order_by(
                        KnowledgeChunk.embedding.cosine_distance(q_vec)
                    )
                    if target_sku:
                        vec_stmt = vec_stmt.where(
                            or_(
                                KnowledgeChunk.sku_id == target_sku,
                                KnowledgeChunk.sku_id == "ALL"
                            )
                        )
                    vec_res = await session.execute(vec_stmt.limit(10))
                    vec_chunks = list(vec_res.scalars().all())
                    return {chunk.id: idx + 1 for idx, chunk in enumerate(vec_chunks)}

                # 核心亮点：超过 300ms 立即抛出 TimeoutError，绝不卡死主流程！
                vector_rank_map = await asyncio.wait_for(
                    _fetch_vector_ranks(),
                    timeout=embed_timeout
                )

            except asyncio.TimeoutError:
                degraded = True
                self.degraded_queries += 1
                degrade_reason = f"向量化耗时超过 {int(embed_timeout*1000)}ms，已平滑降级为纯 BM25 检索"
            except Exception as e:
                degraded = True
                self.degraded_queries += 1
                degrade_reason = f"向量服务异常({type(e).__name__})，已平滑降级为纯 BM25 检索"

            # 4. RRF (倒数排序融合) 合并两路召回结果
            rrf_k = 60
            scored_chunks = []
            for chunk in candidate_chunks:
                score = 0.0
                # 加上 BM25 贡献分
                if chunk.id in bm25_rank_map:
                    score += 1.0 / (rrf_k + bm25_rank_map[chunk.id])
                # 如果没有触发降级，加上向量检索贡献分
                if not degraded and chunk.id in vector_rank_map:
                    score += 1.0 / (rrf_k + vector_rank_map[chunk.id])
                scored_chunks.append((chunk, score))

            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = scored_chunks[:top_k]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        degrade_rate = round((self.degraded_queries / self.total_queries) * 100, 1)

        return {
            "query": query,
            "filtered_sku": target_sku or "未指定(全库检索)",
            "degraded": degraded,
            "degrade_reason": degrade_reason,
            "degrade_rate": f"{degrade_rate}%",
            "latency_ms": latency_ms,
            "results": [
                {
                    "id": c.id,
                    "sku_id": c.sku_id,
                    "category": c.category,
                    "rrf_score": round(s, 4),
                    "content": c.content
                }
                for c, s in top_chunks
            ]
        }


rag_engine = HybridRAGService()


# 本地自测入口：同时测试“正常混合检索”和“超时平滑降级”
async def _test_rag():
    print("=== 测试 1：正常混合检索（带 SKU 硬过滤） ===")
    res1 = await rag_engine.search("How do I reset my SKU-WT01 headphones?", embed_timeout=3.0)
    print(f"耗时: {res1['latency_ms']}ms | 过滤SKU: {res1['filtered_sku']} | 是否降级: {res1['degraded']}")
    print(f"Top-1 召回结果: {res1['results'][0]['content']}\n")

    print("=== 测试 2：模拟网络抖动（严格 300ms 超时平滑降级） ===")
    res2 = await rag_engine.search(
        "What is the return policy for damaged items?",
        embed_timeout=0.3,
        simulate_slow_embed=True
    )
    print(f"耗时: {res2['latency_ms']}ms | 是否降级: {res2['degraded']} | 原因: {res2['degrade_reason']}")
    print(f"当前降级触发率: {res2['degrade_rate']}")
    print(f"降级后 Top-1 依然精准召回: {res2['results'][0]['content']}")


if __name__ == "__main__":
    asyncio.run(_test_rag())