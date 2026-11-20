from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

# 异步连接串：对应刚才 docker-compose.yml 里的账号密码和库名
DATABASE_URL = "postgresql+asyncpg://admin:password123@localhost:5432/cross_border_cs"

# 创建异步连接池，支持高并发访问不卡死
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=20, max_overflow=10)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def init_db():
    async with engine.begin() as conn:
        # 1. 让 PostgreSQL 开启向量检索插件
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # 2. 根据我们写的 Python 类自动在数据库里建表
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session