import json
import re
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.rag_service import rag_engine

router = APIRouter(prefix="/api/chat", tags=["文本智能客服"])

llm_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

# 【简历核心亮点：前缀 Prompt 缓存复用 (Prefix Cache)】
STATIC_SYSTEM_PROMPT = """你是一位专业的跨境电商多语种高级智能客服代表（AI Customer Support Specialist）。
请严格遵守以下服务守则：
1. 语种严格一致：必须使用与买家当前提问完全相同的语言回复（买家说中文，必须全中文回复；买家说英文，必须全英文回复），严禁被英文参考资料带偏语种。
2. 事实绝对忠实：只能根据下方【检索到的知识库参考】回答商品参数、退换货和物流政策。若参考资料未提及，礼貌告知已转接人工专员核实，严禁编造。
3. 语气与风格：热情、专业、简洁高效，直接解决买家痛点。"""

GREETING_WORDS = {"你好", "您好", "在吗", "在不在", "hi", "hello", "hey", "哈喽"}


def detect_target_lang(text: str) -> str:
    """检测买家当前提问语种，防止被英文知识库带偏"""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "简体中文"
    return "与买家提问一致的外语（如英语）"


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"
    simulate_slow_embed: bool = False


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        req_start = time.perf_counter()
        clean_query = req.query.strip().lower().rstrip("！!？?。.")

        # 1. 前置意图分流：纯寒暄直接跳过 RAG 检索，实现 0ms 检索延迟
        if clean_query in GREETING_WORDS:
            rag_res = {
                "filtered_sku": "寒暄免检索",
                "degraded": False,
                "degrade_reason": "",
                "latency_ms": 0.0,
                "results": []
            }
        else:
            # 适配多模态向量模型(doubao-embedding-vision)正常网络往返，正常阈值设为 1.2s
            rag_res = await rag_engine.search(
                query=req.query,
                top_k=3,
                embed_timeout=1.2,
                simulate_slow_embed=req.simulate_slow_embed
            )

        meta_info = {
            "type": "meta",
            "filtered_sku": rag_res["filtered_sku"],
            "degraded": rag_res["degraded"],
            "degrade_reason": rag_res["degrade_reason"],
            "rag_latency_ms": rag_res["latency_ms"]
        }
        yield f"data: {json.dumps(meta_info, ensure_ascii=False)}\n\n"

        # 2. 组装动态上下文 + 语种强制锚定指令
        context_text = "\n".join(
            [f"- [{item['sku_id']} | {item['category']}] {item['content']}" for item in rag_res["results"]]
        )
        target_lang = detect_target_lang(req.query)

        dynamic_user_message = f"""【检索到的知识库参考】
{context_text if context_text else "当前为常规问候或暂无匹配知识"}

【买家当前提问】
{req.query}

【输出语种强制要求】
请务必使用【{target_lang}】回复买家！"""

        messages = [
            {"role": "system", "content": STATIC_SYSTEM_PROMPT},
            {"role": "user", "content": dynamic_user_message}
        ]

        # 3. 流式调用豆包大模型（关闭深度思考，追求极致首字响应）
        first_token_ms = None
        try:
            stream = await llm_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                stream=True,
                temperature=0.3,
                extra_body={"thinking": {"type": "disabled"}}
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - req_start) * 1000, 1)
                    token_text = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'token', 'content': token_text}, ensure_ascii=False)}\n\n"

            done_info = {
                "type": "done",
                "first_token_ms": first_token_ms or 0
            }
            yield f"data: {json.dumps(done_info, ensure_ascii=False)}\n\n"

        except Exception as e:
            err_info = {"type": "error", "message": f"大模型调用异常: {str(e)}"}
            yield f"data: {json.dumps(err_info, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")