import json
import re
import time
from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.rag_service import rag_engine
from app.tools.order_tools import OPENAI_TOOLS_SCHEMA, execute_tool_call

llm_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

AGENT_SYSTEM_PROMPT = """你是一位集“售前咨询”与“售后决策”于一体的跨境电商全能智能客服 Agent。
请严格遵守以下执行规范：
1. 语种严格一致：必须使用与买家当前提问完全相同的语言回复（买家说中文，必须全中文回复；买家说英文，必须全英文回复）。
2. 善用业务工具：
   - 当买家询问订单物流、状态或要求退款/退货时，必须先调用 `query_order_status` 核实订单真实情况，再根据情况调用 `process_refund_request` 办理退款。
   - 严禁在未调用工具的情况下编造订单状态或私自承诺退款！
   - 若买家要求退款但未提供订单号（如 ORD-1001），请礼貌引导买家提供订单号。
3. 知识库忠实：当买家咨询商品参数、通用售后政策时，严格依据【检索到的知识库参考】作答。
4. 风控合规：当工具返回 `HUMAN_APPROVAL_REQUIRED`（金额超限需人工审批）时，明确告知买家生成的审批工单号，切勿承诺已退款。"""

GREETING_WORDS = {"你好", "您好", "在吗", "在不在", "hi", "hello", "hey", "哈喽"}


def detect_target_lang(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "简体中文"
    return "与买家提问一致的外语（如英语）"


async def run_agent_stream(query: str, simulate_slow_embed: bool = False) -> AsyncGenerator[str, None]:
    """统一 Agent 执行引擎：融合混合 RAG 检索 + 多轮 Function Calling 工具链 + SSE 流式输出"""
    req_start = time.perf_counter()
    clean_query = query.strip().lower().rstrip("！!？?。.")

    # 1. 如果问题中不包含纯订单查询号且不是纯寒暄，执行混合 RAG 检索补充商品/政策背景
    has_order_id = bool(re.search(r"ORD-\d+", query.upper()))
    if clean_query in GREETING_WORDS or has_order_id:
        rag_res = {
            "filtered_sku": "订单工具直连" if has_order_id else "寒暄免检索",
            "degraded": False,
            "degrade_reason": "",
            "latency_ms": 0.0,
            "results": []
        }
    else:
        rag_res = await rag_engine.search(
            query=query,
            top_k=3,
            embed_timeout=1.2,
            simulate_slow_embed=simulate_slow_embed
        )

    yield f"data: {json.dumps({'type': 'meta', 'filtered_sku': rag_res['filtered_sku'], 'degraded': rag_res['degraded'], 'degrade_reason': rag_res['degrade_reason'], 'rag_latency_ms': rag_res['latency_ms']}, ensure_ascii=False)}\n\n"

    context_text = "\n".join(
        [f"- [{item['sku_id']} | {item['category']}] {item['content']}" for item in rag_res["results"]]
    )
    target_lang = detect_target_lang(query)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【检索到的知识库参考】\n{context_text if context_text else '无'}\n\n【买家当前提问】\n{query}\n\n【输出语种强制要求】请务必使用【{target_lang}】回复买家！"
        }
    ]

    # 2. Agent ReAct 工具推理循环（最多允许连续调用 3 步工具，如：先查单 -> 再执行退款）
    first_token_ms = None
    try:
        for _step in range(3):
            response = await llm_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                tools=OPENAI_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
                extra_body={"thinking": {"type": "disabled"}}
            )

            msg = response.choices[0].message

            # 如果模型决定调用业务工具（Function Calling）
            if msg.tool_calls:
                messages.append(msg)  # 把模型的 tool_calls 意图记录进对话历史
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments

                    # 实时向前端推送工具调用轨迹卡片（面试展示极佳！）
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': fn_name, 'args': fn_args}, ensure_ascii=False)}\n\n"

                    tool_result_str = await execute_tool_call(fn_name, fn_args)

                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': fn_name, 'result': json.loads(tool_result_str)}, ensure_ascii=False)}\n\n"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_str
                    })
                # 工具执行完，带着工具返回的真实数据库结果进入下一轮循环让模型总结回复
                continue

            # 如果模型不再调用工具，直接流式输出最终回复给买家
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
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.choices[0].delta.content}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'first_token_ms': first_token_ms or 0}, ensure_ascii=False)}\n\n"
            break

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'Agent 执行异常: {str(e)}'}, ensure_ascii=False)}\n\n"