"""crai/agent/main_agent.py — Orquestrador do agente de churn involuntário."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .workflow import (
    diagnose_failure, check_anomaly, infer_payday,
    schedule_retry, trigger_dunning, update_roi_dashboard,
)


def route_after_diagnosis(state: AgentState) -> str:
    """Decide se vale a pena continuar com base no e-Profit."""
    eprofit = state.get("eprofit", 0)
    recommend = state.get("recommend_action", True)
    score = state.get("recovery_score", 0)

    if not recommend or eprofit <= 0:
        print(f"[ROUTER] e-Profit R$ {eprofit:.2f} <= 0 -- abortando (nao vale intervir)")
        return "update_dashboard"
    if score < 5:
        print(f"[ROUTER] Score {score}/100 muito baixo -- abortando")
        return "update_dashboard"
    return "check_anomaly"


def route_after_retry(state: AgentState) -> str:
    if state.get("retry_exhausted") and not state.get("recovered"):
        return "trigger_dunning"
    return "update_dashboard"


def build_crai_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("diagnose", diagnose_failure)
    graph.add_node("check_anomaly", check_anomaly)
    graph.add_node("infer_payday", infer_payday)
    graph.add_node("schedule_retry", schedule_retry)
    graph.add_node("trigger_dunning", trigger_dunning)
    graph.add_node("update_dashboard", update_roi_dashboard)

    graph.set_entry_point("diagnose")
    graph.add_conditional_edges("diagnose", route_after_diagnosis, {
        "check_anomaly": "check_anomaly", "update_dashboard": "update_dashboard",
    })
    graph.add_edge("check_anomaly", "infer_payday")
    graph.add_edge("infer_payday", "schedule_retry")
    graph.add_conditional_edges("schedule_retry", route_after_retry, {
        "trigger_dunning": "trigger_dunning", "update_dashboard": "update_dashboard",
    })
    graph.add_edge("trigger_dunning", "update_dashboard")
    graph.add_edge("update_dashboard", END)

    return graph.compile(checkpointer=MemorySaver())


crai_agent = build_crai_graph()
