# 跨境电商多语种智能客服中枢 (Cross-Border AI Customer Service)

基于 **FastAPI + PostgreSQL (pgvector) + Doubao LLM** 构建的高并发多语种智能客服系统，支持文本 SSE 流式输出与 WebSocket 全双工实时语音打断。

## 核心技术亮点
- **知识库 ETL 与指纹去重**：采用 `SimHash` 计算文本 64 位指纹及海明距离（≤5），入库前过滤高度重复话术。
- **混合检索与 300ms 超时平滑降级**：结合 `pgvector (HNSW)` 向量检索与 `BM25` 稀疏检索（RRF 融合），支持 SKU 前置元数据硬过滤；当外部向量接口超时或异常时，自动平滑回退至纯 BM25 检索，保障服务高可用。
- **首字延迟（TTFT）与缓存优化**：固定静态 System Prompt 触发大模型网关 `Prefix Cache`，禁用非必要思维链推理，复用全局异步 HTTP 连接池。
- **全双工语音交互与毫秒级打断（Barge-in）**：基于原生 `WebSocket` 与 `asyncio.Task` 状态机实现句子级流式 TTS 推流，支持用户插话时毫秒级取消推理任务与语音队列。

## 快速启动
1. 启动数据库：`docker-compose up -d`
2. 配置环境变量：复制 `.env.example` 为 `.env` 并填入 API Key
3. 初始化知识库：`python -m app.services.ingest_service`
4. 启动后端服务：`uvicorn app.main:app --reload`
5. 访问演示控制台：`http://127.0.0.1:8000`