import json
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, select
from app.core.database import Base, AsyncSessionLocal


# 1. 跨境电商订单表
class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(32), primary_key=True)          # 例如 ORD-1001
    user_name = Column(String(64), nullable=False)
    sku_id = Column(String(32), nullable=False)
    product_name = Column(String(128), nullable=False)
    amount_usd = Column(Float, nullable=False)               # 订单实付金额（美元）
    shipping_status = Column(String(32), nullable=False)     # DELIVERED / IN_TRANSIT
    days_since_delivery = Column(Integer, default=0)         # 签收天数
    refund_status = Column(String(32), default="NONE")       # NONE / PENDING_APPROVAL / REFUNDED / REJECTED


# 2. 售后退款审批工单表（Human-in-the-Loop 高危拦截记录）
class RefundTicket(Base):
    __tablename__ = "refund_tickets"

    ticket_id = Column(String(32), primary_key=True)
    order_id = Column(String(32), nullable=False, index=True)
    amount_usd = Column(Float, nullable=False)
    reason = Column(String(256), nullable=False)
    status = Column(String(32), default="PENDING_APPROVAL")  # PENDING_APPROVAL / APPROVED / REJECTED
    created_at = Column(DateTime, default=datetime.utcnow)


async def seed_mock_orders():
    """初始化 3 笔典型跨境测试订单（小额已签收、大额高危订单、运输中订单）"""
    async with AsyncSessionLocal() as session:
        existing = await session.execute(select(Order).limit(1))
        if existing.scalar_one_or_none():
            return

        mock_orders = [
            Order(
                order_id="ORD-1001",
                user_name="Alice",
                sku_id="SKU-WT01",
                product_name="SKU-WT01 Pro Wireless Headphones",
                amount_usd=39.9,
                shipping_status="DELIVERED",
                days_since_delivery=5,
                refund_status="NONE"
            ),
            Order(
                order_id="ORD-1002",
                user_name="Bob",
                sku_id="SKU-SW09",
                product_name="SKU-SW09 Ultra Smart Watch",
                amount_usd=189.0,  # 超过 $50 风控阈值，将触发人工审批拦截！
                shipping_status="DELIVERED",
                days_since_delivery=3,
                refund_status="NONE"
            ),
            Order(
                order_id="ORD-1003",
                user_name="Carlos",
                sku_id="SKU-WT01",
                product_name="SKU-WT01 Pro Wireless Headphones",
                amount_usd=39.9,
                shipping_status="IN_TRANSIT",
                days_since_delivery=0,
                refund_status="NONE"
            ),
        ]
        session.add_all(mock_orders)
        await session.commit()
        print("✅ 跨境模拟订单数据 (ORD-1001 / ORD-1002 / ORD-1003) 初始化完成！")


# ================= 给大模型调用的标准化 Function Calling 工具集 =================

OPENAI_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_order_status",
            "description": "根据订单号查询跨境电商订单的商品详情、实付金额、物流状态及签收天数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如 ORD-1001"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund_request",
            "description": "为指定订单发起售后退款申请。系统会自动校验退货时效、幂等性（防重复退款）以及风控金额阈值（>$50 需转人工审批）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "需要申请退款的订单编号，例如 ORD-1001"
                    },
                    "reason": {
                        "type": "string",
                        "description": "买家的退款原因简述"
                    }
                },
                "required": ["order_id", "reason"]
            }
        }
    }
]


async def execute_tool_call(tool_name: str, arguments_json: str) -> str:
    """安全路由并执行大模型发起的工具调用"""
    args = json.loads(arguments_json)
    order_id = args.get("order_id", "").strip().upper()

    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Order).where(Order.order_id == order_id))
        order = res.scalar_one_or_none()

        if not order:
            return json.dumps({"success": False, "error": f"未查询到订单号 {order_id}，请核对订单号是否正确。"}, ensure_ascii=False)

        if tool_name == "query_order_status":
            return json.dumps({
                "success": True,
                "order_id": order.order_id,
                "product": order.product_name,
                "amount_usd": order.amount_usd,
                "shipping_status": order.shipping_status,
                "days_since_delivery": order.days_since_delivery,
                "refund_status": order.refund_status
            }, ensure_ascii=False)

        elif tool_name == "process_refund_request":
            reason = args.get("reason", "买家申请售后")

            # 1. 幂等性校验：防止重复申请退款
            if order.refund_status in ("REFUNDED", "PENDING_APPROVAL"):
                return json.dumps({
                    "success": False,
                    "status": order.refund_status,
                    "message": f"订单 {order_id} 当前已处于【{order.refund_status}】状态，请勿重复提交退款申请。"
                }, ensure_ascii=False)

            # 2. 退货政策硬校验：超过 30 天不可自动退款
            if order.shipping_status == "DELIVERED" and order.days_since_delivery > 30:
                return json.dumps({
                    "success": False,
                    "status": "OUT_OF_POLICY",
                    "message": f"订单 {order_id} 已签收 {order.days_since_delivery} 天，超过 30 天无理由退货期限。"
                }, ensure_ascii=False)

            # 3. 风控熔断拦截（Human-in-the-Loop）：金额 > $50 挂起转人工审批！
            if order.amount_usd > 50.0:
                ticket_id = f"TK-{uuid.uuid4().hex[:6].upper()}"
                order.refund_status = "PENDING_APPROVAL"
                ticket = RefundTicket(
                    ticket_id=ticket_id,
                    order_id=order_id,
                    amount_usd=order.amount_usd,
                    reason=reason,
                    status="PENDING_APPROVAL"
                )
                session.add(ticket)
                await session.commit()
                return json.dumps({
                    "success": True,
                    "action": "HUMAN_APPROVAL_REQUIRED",
                    "ticket_id": ticket_id,
                    "amount_usd": order.amount_usd,
                    "message": f"由于该订单退款金额为 ${order.amount_usd}（超过 $50 风控自动审批上限），系统已生成人工审核工单 {ticket_id} 并挂起，请安抚买家等待专员审核（预计 2 小时内处理）。"
                }, ensure_ascii=False)

            # 4. 符合自动退款条件（<= $50）：秒级自动执行退款落库
            order.refund_status = "REFUNDED"
            await session.commit()
            return json.dumps({
                "success": True,
                "action": "AUTO_REFUNDED",
                "amount_usd": order.amount_usd,
                "message": f"订单 {order_id} 符合极速自动退款条件（金额 ${order.amount_usd} <= $50），已为您自动办理全额原路退回。"
            }, ensure_ascii=False)

    return json.dumps({"success": False, "error": "未知工具调用"}, ensure_ascii=False)