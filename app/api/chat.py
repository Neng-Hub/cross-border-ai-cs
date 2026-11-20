from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.services.agent_service import run_agent_stream
from app.tools.order_tools import Order, RefundTicket

router = APIRouter(prefix="/api/chat", tags=["全能智能客服与售后工作台"])


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"
    simulate_slow_embed: bool = False


class ApprovalRequest(BaseModel):
    action: str  # "APPROVE" 或 "REJECT"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """统一流式对话接口：自动路由售前 RAG 问答与售后多步工具调用"""
    return StreamingResponse(
        run_agent_stream(query=req.query, simulate_slow_embed=req.simulate_slow_embed),
        media_type="text/event-stream"
    )


@router.get("/dashboard")
async def get_dashboard_data():
    """供前端工作台实时展示当前订单表与待人工审核的售后工单"""
    async with AsyncSessionLocal() as session:
        orders_res = await session.execute(select(Order).order_by(Order.order_id))
        tickets_res = await session.execute(select(RefundTicket).order_by(RefundTicket.created_at.desc()))

        orders = [
            {
                "order_id": o.order_id,
                "user_name": o.user_name,
                "product": o.product_name,
                "amount_usd": o.amount_usd,
                "shipping_status": o.shipping_status,
                "days": o.days_since_delivery,
                "refund_status": o.refund_status
            }
            for o in orders_res.scalars().all()
        ]
        tickets = [
            {
                "ticket_id": t.ticket_id,
                "order_id": t.order_id,
                "amount_usd": t.amount_usd,
                "reason": t.reason,
                "status": t.status
            }
            for t in tickets_res.scalars().all()
        ]
        return {"orders": orders, "tickets": tickets}


@router.post("/tickets/{ticket_id}/review")
async def review_refund_ticket(ticket_id: str, req: ApprovalRequest):
    """Human-in-the-Loop 人工专员审批高危退款工单"""
    async with AsyncSessionLocal() as session:
        t_res = await session.execute(select(RefundTicket).where(RefundTicket.ticket_id == ticket_id))
        ticket = t_res.scalar_one_or_none()
        if not ticket:
            raise HTTPException(status_code=404, detail="工单不存在")

        o_res = await session.execute(select(Order).where(Order.order_id == ticket.order_id))
        order = o_res.scalar_one_or_none()

        if req.action.upper() == "APPROVE":
            ticket.status = "APPROVED"
            if order:
                order.refund_status = "REFUNDED"
        else:
            ticket.status = "REJECTED"
            if order:
                order.refund_status = "REJECTED"

        await session.commit()
        return {"success": True, "ticket_id": ticket_id, "new_status": ticket.status}


@router.post("/reset-demo")
async def reset_demo_orders():
    """一键重置演示订单状态，方便反复演示面试场景"""
    async with AsyncSessionLocal() as session:
        o_res = await session.execute(select(Order))
        for o in o_res.scalars().all():
            o.refund_status = "NONE"
        t_res = await session.execute(select(RefundTicket))
        for t in t_res.scalars().all():
            await session.delete(t)
        await session.commit()
        return {"success": True}