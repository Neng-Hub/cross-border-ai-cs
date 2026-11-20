from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True)
    # 业务元数据：用于检索前的 WHERE 硬过滤
    sku_id = Column(String(64), index=True, nullable=True, comment="商品SKU，通用政策为ALL")
    category = Column(String(32), index=True, nullable=False, comment="分类：logistics/return/product")
    language = Column(String(16), default="en", index=True, comment="语种：en/zh/es")
    
    # 文本内容与指纹去重字段
    content = Column(Text, nullable=False, comment="切片文本内容")
    simhash_fingerprint = Column(String(64), index=True, nullable=False, comment="SimHash文本指纹，用于去重")
    
    # 向量字段（先按主流的 1536 维设置，后面换模型可调）
    embedding = Column(Vector(1536), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 为向量检索创建 HNSW 索引（提升百万级数据下的检索速度）
    __table_args__ = (
        Index(
            "ix_knowledge_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )