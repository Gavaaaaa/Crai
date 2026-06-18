"""
crai/churn_voluntary/risk_scorer.py
Calcula o risk_score a partir de eventos do Segment SDK.

Eventos suportados:
    Cancellation Page Viewed  → risco fixo alto (intenção explícita)
    Downgrade Clicked         → risco fixo médio-alto
    Session Started           → risco calculado por inatividade + uso
"""

FIXED_RISK = {
    "Cancellation Page Viewed": 0.90,
    "Downgrade Clicked":        0.75,
}


def calculate_risk(event: str, props: dict) -> float:
    if event in FIXED_RISK:
        return FIXED_RISK[event]

    if event == "Session Started":
        days = props.get("days_since_last", 0)
        features = props.get("features_used_30d", 10)
        # Risco sobe com inatividade, cai com uso de features
        risk = min(1.0, (days / 30) * 0.7 + max(0, (5 - features) / 5) * 0.3)
        return round(risk, 3)

    return 0.0   # evento desconhecido — não dispara nada


def classify_profile(props: dict) -> str:
    """Reaproveita o mesmo critério do churn involuntário (CV de pagamentos)."""
    return props.get("billing_profile", "CLT")
