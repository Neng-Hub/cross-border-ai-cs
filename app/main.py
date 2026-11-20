from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.core.database import init_db
from app.models import knowledge
from app.tools import order_tools
from app.tools.order_tools import seed_mock_orders
from app.api.chat import router as chat_router
from app.api.voice import router as voice_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_mock_orders()
    print("✅ PostgreSQL 连接成功，知识库表、订单表与售后工单表已全部就绪！")
    yield

app = FastAPI(title="跨境电商全能智能客服与售后决策中枢", lifespan=lifespan)

app.include_router(chat_router)
app.include_router(voice_router)

@app.get("/")
async def serve_frontend():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok"}