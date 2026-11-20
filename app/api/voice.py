import asyncio
import json
import re
import time
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.rag_service import rag_engine

router = APIRouter(prefix="/ws", tags=["全双工实时语音客服"])

llm_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

# 语音交互专用的简短口语化 System Prompt（同样放在最前面触发 Prefix Cache）
VOICE_SYSTEM_PROMPT = """你是一位跨境电商多语种实时语音智能客服。
由于当前处于电话/语音通话场景，请严格遵守以下规则：
1. 语种严格一致：买家说中文，必须全中文回复；买家说英文，必须全英文回复。
2. 适合语音播报：回答必须口语化、自然简短（控制在 2~3 句话以内），绝对不要输出 Markdown 符号（如 **、#、列表横杠）或复杂网址，方便语音直接朗读。
3. 事实忠实：只能依据【检索到的知识库参考】回答，未知信息礼貌告知转接人工。"""

GREETING_WORDS = {"你好", "您好", "在吗", "hi", "hello", "hey", "哈喽"}


def detect_lang_code(text: str) -> str:
    """识别语种，返回给前端用于切换对应的 TTS 语音包"""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-CN"
    return "en-US"


@router.websocket("/voice")
async def voice_full_duplex_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_gen_task: Optional[asyncio.Task] = None
    turn_id = 0  # 对话轮次版本号，防止旧轮次残留数据包串台

    async def cancel_current_generation(reason: str = "用户语音打断"):
        """【核心亮点：毫秒级打断 (Barge-in)】立即取消正在进行的 RAG 检索与大模型流式生成"""
        nonlocal current_gen_task
        if current_gen_task and not current_gen_task.done():
            t0 = time.perf_counter()
            current_gen_task.cancel()
            try:
                await current_gen_task
            except asyncio.CancelledError:
                pass
            cancel_ms = round((time.perf_counter() - t0) * 1000, 2)
            await websocket.send_json({
                "type": "interrupted",
                "reason": reason,
                "cancel_latency_ms": cancel_ms
            })

    async def stream_voice_response(user_text: str, active_turn: int):
        """后台异步流水线：RAG 检索 -> 大模型流式生成 -> 按标点切分句子推送给前端 TTS"""
        req_start = time.perf_counter()
        lang_code = detect_lang_code(user_text)
        clean_q = user_text.strip().lower().rstrip("！!？?。.")

        try:
            # 1. 检索阶段
            if clean_q in GREETING_WORDS:
                rag_res = {"results": [], "latency_ms": 0.0, "degraded": False}
            else:
                rag_res = await rag_engine.search(query=user_text, top_k=2, embed_timeout=1.2)

            if active_turn != turn_id:
                return

            context_text = "\n".join([f"- {item['content']}" for item in rag_res["results"]])
            lang_hint = "简体中文" if lang_code == "zh-CN" else "英语"

            messages = [
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"【参考资料】\n{context_text or '常规问候'}\n\n【买家语音转写】\n{user_text}\n\n请用【{lang_hint}】简短口语化回复（不要带任何星号或特殊符号）："
                }
            ]

            # 2. 流式调用豆包大模型，按句号/问号/感叹号实时切分“首句”，实现极低首包语音播报延迟
            stream = await llm_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                stream=True,
                temperature=0.3,
                extra_body={"thinking": {"type": "disabled"}}
            )

            sentence_buffer = ""
            first_chunk_sent = False
            sentence_delimiters = ("。", "！", "？", ".", "!", "?", "\n")

            async for chunk in stream:
                if active_turn != turn_id:
                    return
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    sentence_buffer += token

                    # 推送实时字幕流
                    await websocket.send_json({
                        "type": "subtitle_token",
                        "turn_id": active_turn,
                        "token": token
                    })

                    # 一旦凑满一句完整短句，立刻推送给前端 TTS 引擎开始发声，不用等整段话生成完！
                    if any(sentence_buffer.endswith(d) for d in sentence_delimiters) and len(sentence_buffer.strip()) >= 2:
                        first_audio_ms = round((time.perf_counter() - req_start) * 1000, 1) if not first_chunk_sent else None
                        first_chunk_sent = True
                        await websocket.send_json({
                            "type": "tts_sentence",
                            "turn_id": active_turn,
                            "text": sentence_buffer.strip(),
                            "lang": lang_code,
                            "first_sentence_ms": first_audio_ms
                        })
                        sentence_buffer = ""

            # 推送末尾剩余短句
            if sentence_buffer.strip() and active_turn == turn_id:
                first_audio_ms = round((time.perf_counter() - req_start) * 1000, 1) if not first_chunk_sent else None
                await websocket.send_json({
                    "type": "tts_sentence",
                    "turn_id": active_turn,
                    "text": sentence_buffer.strip(),
                    "lang": lang_code,
                    "first_sentence_ms": first_audio_ms
                })

            await websocket.send_json({"type": "turn_done", "turn_id": active_turn})

        except asyncio.CancelledError:
            # 被打断时安静退出协程，释放网络连接
            raise
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            event_type = data.get("type")

            if event_type == "interrupt":
                # 前端检测到用户开始说话（VAD 触发）或点击打断按钮 -> 立即掐断当前 AI 任务
                turn_id += 1
                await cancel_current_generation("前端检测到用户插话 (Barge-in)")

            elif event_type == "user_speech":
                # 收到用户完整语音识别文本 -> 先打断可能残留的旧任务，再启动新一轮异步生成
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue
                turn_id += 1
                await cancel_current_generation("新语音输入抢占")
                current_gen_task = asyncio.create_task(stream_voice_response(user_text, turn_id))

    except WebSocketDisconnect:
        if current_gen_task and not current_gen_task.done():
            current_gen_task.cancel()