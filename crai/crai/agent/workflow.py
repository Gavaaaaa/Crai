"""crai/agent/workflow.py — Nós do pipeline de churn involuntário."""

import numpy as np
from datetime import datetime
from .state import AgentState
from ..ml.failure_classifier import FailureClassifier
from ..ml.anomaly_detector import AnomalyDetector
from ..ml.payday_inference import PaydayInference
from ..dunning.smart_backoff import SmartBackoff
from ..dunning.dunning_engine import DunningEngine
from ..integrations.hubspot_crm import HubSpotCRM

_classifier = FailureClassifier()
_detector   = AnomalyDetector()
_payday     = PaydayInference()
_backoff    = SmartBackoff()
_dunning    = DunningEngine()
_hubspot    = HubSpotCRM()

# Tentar carregar modelos treinados; se não existirem, usa heurística
_classifier.load()
_detector.load()
_payday.load()


async def diagnose_failure(state: AgentState) -> AgentState:
    """Diagnostica causa da falha via ensemble XGBoost+RF com e-Profit e SHAP."""
    features = _extract_features(state["stripe_event"], state["amount"])
    result = _classifier.predict(features)

    shap_readable = result["shap_explanation"].get("readable", "")
    print(f"[AGENT] Diagnóstico ({result['method']}): "
          f"score {result['recovery_score']}/100 | "
          f"e-Profit R$ {result['eprofit']:.2f} | "
          f"ação: {'SIM' if result['recommend_action'] else 'NÃO'}")
    if shap_readable:
        print(f"[SHAP]  {shap_readable}")

    return {
        **state,
        "failure_cause": features["gateway_error_code"],
        "recovery_score": result["recovery_score"],
        "p_recovery": result["p_recovery"],
        "eprofit": result["eprofit"],
        "recommend_action": result["recommend_action"],
        "ltv_estimated": result["ltv_estimated"],
        "shap_explanation": result["shap_explanation"],
        "feature_importance": {
            f["feature"]: f["contribution_pct"]
            for f in result["shap_explanation"].get("features", [])[:5]
        },
    }


async def check_anomaly(state: AgentState) -> AgentState:
    result = await _detector.check(state["customer_id"], state["stripe_event"])

    # Anomalia ajusta o score de recuperação para baixo
    if result["is_anomaly"]:
        adjusted_score = max(0, int(state["recovery_score"] * 0.7))
        adjusted_p = state.get("p_recovery", 0.5) * 0.7
    else:
        adjusted_score = state["recovery_score"]
        adjusted_p = state.get("p_recovery", 0.5)

    # Recalcular e-Profit com score ajustado
    ltv = state.get("ltv_estimated", state["amount"] * 6)
    cost = 0.05  # bot_whatsapp padrão
    new_eprofit = round(float(adjusted_p * ltv - cost), 2)

    print(f"[AGENT] Anomalia ({result['method']}): {result['is_anomaly']} | "
          f"erro: {result['error']:.4f} | threshold: {result['threshold']:.4f}")
    if result["is_anomaly"] and result.get("top_features"):
        top = ", ".join(f["feature"] for f in result["top_features"])
        print(f"[AGENT] Features anômalas: {top}")

    return {
        **state,
        "is_anomalous": result["is_anomaly"],
        "reconstruction_error": result["error"],
        "anomaly_explanation": result.get("top_features", []),
        "recovery_score": adjusted_score,
        "p_recovery": round(adjusted_p, 4),
        "eprofit": new_eprofit,
        "recommend_action": bool(new_eprofit > 0),
    }


async def infer_payday(state: AgentState) -> AgentState:
    if state["failure_cause"] != "insufficient_funds":
        return {**state, "optimal_retry_at": None}
    window = await _payday.predict_next_window(state["customer_id"])
    print(f"[AGENT] Payday ({window.get('method', 'heuristic')}): "
          f"{window['timestamp'].strftime('%d/%m %H:%M')} | perfil: {window['profile']} | "
          f"confiança: {window['confidence']:.0%}")
    return {**state, "optimal_retry_at": window["timestamp"], "confidence": window["confidence"],
            "profile_type": window["profile"]}


async def schedule_retry(state: AgentState) -> AgentState:
    attempt = state.get("retry_count", 0)
    result = _backoff.get_schedule(state["failure_cause"], attempt, state.get("optimal_retry_at"))
    if result["exhausted"]:
        print(f"[AGENT] Retentativas esgotadas")
    else:
        print(f"[AGENT] Retry agendado: {result['schedule_at'].strftime('%d/%m %H:%M')} ({result.get('strategy')})")
    return {**state, "next_retry_at": result.get("schedule_at"), "retry_exhausted": result["exhausted"],
            "retry_count": attempt + 1}


async def trigger_dunning(state: AgentState) -> AgentState:
    p_recovery = state.get("p_recovery", state.get("recovery_score", 50) / 100)
    result = await _dunning.run_campaign(state["customer_id"], state["failure_cause"],
                                          p_recovery, state["amount"])
    return {**state, "dunning_sent": result["sent"], "channel": result["channel"], "message_sent": result["message"]}


async def update_roi_dashboard(state: AgentState) -> AgentState:
    fee = state["amount"] * 0.15 if state.get("recovered") else 0
    eprofit = state.get("eprofit", 0)
    recovered_icon = "[OK]" if state.get("recovered") else "[X]"
    print(f"[ROI] {recovered_icon} R$ {state['amount']:.2f} | taxa R$ {fee:.2f} | e-Profit R$ {eprofit:.2f}")
    crm_result = await _hubspot.register_recovery_cycle(state)
    print(f"[HUBSPOT] Contact {crm_result['hubspot_contact_id']} | Deal {crm_result['hubspot_deal_id']} | {crm_result['stage']}\n")
    return state


def _extract_features(event: dict, amount: float) -> dict:
    """Extrai as 11 features + LTV para o novo classificador."""
    charge = event.get("data", {}).get("object", {})
    now = datetime.now()

    # Simular tenure e histórico (em produção viriam do banco/CRM)
    rng = np.random.default_rng(seed=abs(hash(charge.get("customer", ""))) % (2**32))
    tenure = int(rng.exponential(scale=12))
    payment_history = round(float(np.clip(rng.beta(5, 2), 0, 1)), 3)
    failure_count = int(rng.poisson(1.5))

    invoice_amount = charge.get("amount", 0) / 100 if charge.get("amount", 0) > 100 else amount
    avg_ticket = round(invoice_amount * rng.uniform(0.9, 1.1), 2)
    ltv = round(max(invoice_amount, tenure * avg_ticket * 0.9 / 12), 2)

    return {
        "gateway_error_code": charge.get("failure_code") or "processing_error",
        "card_brand": (charge.get("payment_method_details") or {}).get("brand") or "visa",
        "tenure_months": tenure,
        "day_of_month": now.day,
        "invoice_amount": invoice_amount,
        "avg_ticket": avg_ticket,
        "payment_history_score": payment_history,
        "failure_count_90d": failure_count,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "attempt_count": charge.get("attempt_count", 1),
        "ltv_estimated": ltv,
    }
