from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.core.database import init_db
from app.models import knowledge
from app.api.chat import router as chat_router
from app.api.voice import router as voice_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("✅ PostgreSQL 连接成功，pgvector 扩展与 knowledge_chunks 表已就绪！")
    yield

app = FastAPI(title="跨境电商智能客服系统", lifespan=lifespan)

# 注册文本流式聊天路由 + 全双工实时语音 WebSocket 路由
app.include_router(chat_router)
app.include_router(voice_router)

@app.get("/")
async def serve_frontend():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok"}