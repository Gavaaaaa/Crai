"""crai/agent/workflow.py — Nós do pipeline de churn involuntário."""

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


async def diagnose_failure(state: AgentState) -> AgentState:
    features = _extract_features(state["stripe_event"])
    result = await _classifier.predict(features)
    print(f"[AGENT] Diagnóstico: {result['cause']} | score: {result['score']:.2f} ({result.get('method')})")
    return {**state, "failure_cause": result["cause"], "recovery_score": result["score"],
            "feature_importance": result.get("importance", {})}


async def check_anomaly(state: AgentState) -> AgentState:
    result = await _detector.check(state["customer_id"], state["stripe_event"])
    adjusted = state["recovery_score"] * (0.7 if result["is_anomaly"] else 1.0)
    print(f"[AGENT] Anomalia: {result['is_anomaly']} | erro: {result['error']:.4f}")
    return {**state, "is_anomalous": result["is_anomaly"], "reconstruction_error": result["error"],
            "recovery_score": round(adjusted, 3)}


async def infer_payday(state: AgentState) -> AgentState:
    if state["failure_cause"] != "insufficient_funds":
        return {**state, "optimal_retry_at": None}
    window = await _payday.predict_next_window(state["customer_id"])
    print(f"[AGENT] Payday: {window['timestamp'].strftime('%d/%m %H:%M')} | perfil: {window['profile']}")
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
    result = await _dunning.run_campaign(state["customer_id"], state["failure_cause"],
                                          state["recovery_score"], state["amount"])
    return {**state, "dunning_sent": result["sent"], "channel": result["channel"], "message_sent": result["message"]}


async def update_roi_dashboard(state: AgentState) -> AgentState:
    fee = state["amount"] * 0.15 if state.get("recovered") else 0
    print(f"[ROI] {'✅' if state.get('recovered') else '❌'} R$ {state['amount']:.2f} | taxa R$ {fee:.2f}")
    crm_result = await _hubspot.register_recovery_cycle(state)
    print(f"[HUBSPOT] Contact {crm_result['hubspot_contact_id']} | Deal {crm_result['hubspot_deal_id']} | {crm_result['stage']}\n")
    return state


def _extract_features(event: dict) -> dict:
    charge = event.get("data", {}).get("object", {})
    now = datetime.now()
    return {
        "error_code":   charge.get("failure_code"),
        "amount":       charge.get("amount", 0) / 100,
        "hour_of_day":  now.hour,
        "day_of_week":  now.weekday(),
        "card_brand":   (charge.get("payment_method_details") or {}).get("brand"),
        "decline_code": charge.get("failure_message"),
    }
