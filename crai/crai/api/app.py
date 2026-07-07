"""
crai/api/app.py
FastAPI — unifica os dois pipelines da CRAI:
  /webhooks/stripe        → churn involuntário
  /webhooks/segment       → churn voluntário
  /simulate/*             → endpoints de teste sem precisar do Stripe/Segment reais
"""

import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..agent.main_agent import crai_agent
from ..agent.state import AgentState
from ..churn_voluntary.voluntary_agent import voluntary_churn_agent
from ..churn_voluntary.state import ChurnVoluntaryState

app = FastAPI(title="CRAI", version="2.0.0",
              description="Agente autônomo de recuperação de receita — churn involuntário + voluntário")

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")


# ── Churn Involuntário ───────────────────────────────────────────────────

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request) -> JSONResponse:
    payload = await request.body()
    event = json.loads(payload)  # modo dev: sem validação de assinatura
    if event.get("type") == "invoice.payment_failed":
        await _run_involuntary_pipeline(event)
    return JSONResponse({"status": "ok"})


class SimulatePayment(BaseModel):
    customer_id:  str   = "cus_demo_001"
    amount:       float = 299.90
    failure_code: str   = "insufficient_funds"


@app.post("/simulate/payment-failed")
async def simulate_payment_failed(payload: SimulatePayment) -> JSONResponse:
    event = _build_fake_stripe_event(payload)
    await _run_involuntary_pipeline(event)
    return JSONResponse({"status": "pipeline_executado", "customer_id": payload.customer_id})


# ── Churn Voluntário ─────────────────────────────────────────────────────

@app.post("/webhooks/segment")
async def segment_webhook(request: Request) -> JSONResponse:
    payload = await request.json()
    await _run_voluntary_pipeline(
        user_id=payload.get("userId", "usr_unknown"),
        event=payload.get("event", ""),
        props=payload.get("properties", {}),
    )
    return JSONResponse({"status": "ok"})


class SimulateChurnRisk(BaseModel):
    user_id: str = "usr_demo_001"
    event:   str = "Cancellation Page Viewed"   # ou "Downgrade Clicked" | "Session Started"
    days_since_last:   int = 14
    features_used_30d: int = 2
    on_site_now: bool = True
    billing_profile: str = "CLT"


@app.post("/simulate/churn-risk")
async def simulate_churn_risk(payload: SimulateChurnRisk) -> JSONResponse:
    props = {
        "days_since_last":   payload.days_since_last,
        "features_used_30d": payload.features_used_30d,
        "on_site_now":       payload.on_site_now,
        "billing_profile":   payload.billing_profile,
    }
    await _run_voluntary_pipeline(payload.user_id, payload.event, props)
    return JSONResponse({"status": "pipeline_executado", "user_id": payload.user_id})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "crai-agent-v2"}


# ── Helpers ────────────────────────────────────────────────────────────────

async def _run_involuntary_pipeline(event: dict):
    invoice = event.get("data", {}).get("object", {})
    customer_id = invoice.get("customer", "cus_unknown")
    amount = invoice.get("amount_due", 0) / 100

    initial: AgentState = {
        "stripe_event": event, "customer_id": customer_id,
        "invoice_id": invoice.get("id", "inv_unknown"), "amount": amount,
        "failure_cause": None, "recovery_score": None, "p_recovery": None,
        "eprofit": None, "recommend_action": None, "ltv_estimated": None,
        "shap_explanation": None, "feature_importance": None,
        "is_anomalous": None, "reconstruction_error": None, "anomaly_explanation": None,
        "optimal_retry_at": None,
        "confidence": None, "profile_type": None, "retry_count": 0, "next_retry_at": None,
        "retry_exhausted": False, "recovered": False, "dunning_sent": False,
        "channel": None, "message_sent": None,
    }
    config = {"configurable": {"thread_id": customer_id}}
    await crai_agent.ainvoke(initial, config)


async def _run_voluntary_pipeline(user_id: str, event: str, props: dict):
    initial: ChurnVoluntaryState = {
        "user_id": user_id, "event": event, "props": props,
        "risk_score": 0.0, "profile": "CLT", "offer_type": None,
        "channel": None, "on_site_now": props.get("on_site_now", False),
        "prior_channel_success": None, "message": None,
        "offer_sent": False, "accepted": None, "retained": False, "escalated_to_human": False,
    }
    config = {"configurable": {"thread_id": user_id}}
    await voluntary_churn_agent.ainvoke(initial, config)


def _build_fake_stripe_event(p: SimulatePayment) -> dict:
    return {
        "id": f"evt_test_{p.customer_id}", "type": "invoice.payment_failed",
        "data": {"object": {
            "id": f"inv_test_{p.customer_id}", "customer": p.customer_id,
            "amount_due": int(p.amount * 100), "currency": "brl",
            "failure_code": p.failure_code, "failure_message": f"Teste: {p.failure_code}",
            "attempt_count": 1, "payment_method_details": {"brand": "visa", "last4": "4242"},
        }},
    }
