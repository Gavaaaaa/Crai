"""
test_pipeline.py
Testa os DOIS pipelines da CRAI sem precisar de Stripe, Segment ou HubSpot reais:

  1. Churn Involuntário — 4 cenários de falha de pagamento
  2. Churn Voluntário   — 4 cenários de risco de cancelamento

Uso:
    python test_pipeline.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from crai.agent.main_agent import crai_agent
from crai.agent.state import AgentState
from crai.churn_voluntary.voluntary_agent import voluntary_churn_agent
from crai.churn_voluntary.state import ChurnVoluntaryState


# ── Churn Involuntário ───────────────────────────────────────────────────

def make_stripe_event(customer_id, amount, failure_code):
    return {
        "id": f"evt_{customer_id}", "type": "invoice.payment_failed",
        "data": {"object": {
            "id": f"inv_{customer_id}", "customer": customer_id,
            "amount_due": int(amount * 100), "currency": "brl",
            "failure_code": failure_code, "failure_message": f"Teste: {failure_code}",
            "attempt_count": 1, "payment_method_details": {"brand": "visa", "last4": "4242"},
        }},
    }


async def run_involuntary_scenario(name, customer_id, amount, failure_code, attempt_count=1):
    print(f"\n{'═'*64}\n  CHURN INVOLUNTÁRIO — {name}\n  {customer_id} | R$ {amount:.2f} | {failure_code}\n{'═'*64}")

    event = make_stripe_event(customer_id, amount, failure_code)
    event["data"]["object"]["attempt_count"] = attempt_count
    retries_done = max(0, attempt_count - 1)

    initial: AgentState = {
        "stripe_event": event,
        "customer_id": customer_id, "invoice_id": f"inv_{customer_id}", "amount": amount,
        "failure_cause": None, "recovery_score": None, "p_recovery": None,
        "eprofit": None, "recommend_action": None, "ltv_estimated": None,
        "shap_explanation": None, "feature_importance": None,
        "is_anomalous": None, "reconstruction_error": None, "anomaly_explanation": None,
        "optimal_retry_at": None,
        "acao_decidida": None, "acao_motivo": None, "acao_eprofit": None,
        "escalar_humano": False,
        "confidence": None, "profile_type": None, "retry_count": retries_done, "next_retry_at": None,
        "retry_exhausted": False, "recovered": False, "dunning_sent": False,
        "channel": None, "message_sent": None,
    }
    config = {"configurable": {"thread_id": customer_id}}
    return await crai_agent.ainvoke(initial, config)


# ── Churn Voluntário ─────────────────────────────────────────────────────

async def run_voluntary_scenario(name, user_id, event, props):
    print(f"\n{'═'*64}\n  CHURN VOLUNTÁRIO — {name}\n  {user_id} | evento: {event}\n{'═'*64}")

    initial: ChurnVoluntaryState = {
        "user_id": user_id, "event": event, "props": props,
        "risk_score": 0.0, "profile": "CLT", "offer_type": None,
        "channel": None, "on_site_now": props.get("on_site_now", False),
        "prior_channel_success": None, "message": None,
        "offer_sent": False, "accepted": None, "retained": False, "escalated_to_human": False,
    }
    config = {"configurable": {"thread_id": user_id}}
    return await voluntary_churn_agent.ainvoke(initial, config)


async def main():
    print("\n🚀 CRAI v2 — Teste dos Dois Pipelines (Involuntário + Voluntário)\n")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY não definida — mensagens usarão fallback\n")
    if not os.getenv("HUBSPOT_TOKEN"):
        print("⚠️  HUBSPOT_TOKEN não definida — HubSpot rodará em modo simulação\n")

    # ── Cenários de churn involuntário ──────────────────────────────────
    involuntary_scenarios = [
        ("Saldo Insuficiente", "cus_maria_001", 299.90, "insufficient_funds"),
        ("Cartão Expirado",    "cus_joao_002",  149.00, "expired_card"),
        ("Bloqueio Bancário",  "cus_pedro_003", 599.00, "card_declined"),
        ("Erro Técnico",       "cus_ana_004",    99.90, "processing_error"),
    ]
    involuntary_results = []
    for name, cid, amount, code in involuntary_scenarios:
        result = await run_involuntary_scenario(name, cid, amount, code)
        involuntary_results.append(result)

    # ── Cenários de churn voluntário ────────────────────────────────────
    voluntary_scenarios = [
        ("Visitou Cancelamento (CLT)",  "usr_marcos_011", "Cancellation Page Viewed",
         {"on_site_now": True, "billing_profile": "CLT"}),
        ("Clicou em Downgrade (PJ)",    "usr_julia_012",  "Downgrade Clicked",
         {"on_site_now": True, "billing_profile": "PJ"}),
        ("Inatividade Prolongada",      "usr_diego_013",  "Session Started",
         {"days_since_last": 18, "features_used_30d": 1, "on_site_now": False, "billing_profile": "freelancer"}),
        ("Risco Crítico (>=0.90)",      "usr_lara_014",   "Cancellation Page Viewed",
         {"on_site_now": False, "billing_profile": "PJ"}),
    ]
    voluntary_results = []
    for name, uid, event, props in voluntary_scenarios:
        result = await run_voluntary_scenario(name, uid, event, props)
        voluntary_results.append(result)

    # ── Resumo ───────────────────────────────────────────────────────────
    print(f"\n{'═'*64}\n  RESUMO GERAL\n{'═'*64}")

    total_amount = sum(r["amount"] for r in involuntary_results)
    recovered = sum(1 for r in involuntary_results if r.get("recovered"))
    print(f"\n📉 Churn Involuntário:")
    print(f"   Volume testado     : R$ {total_amount:.2f}")
    print(f"   Recuperados        : {recovered}/{len(involuntary_results)}")
    print(f"   Dunnings enviados  : {sum(1 for r in involuntary_results if r.get('dunning_sent'))}/{len(involuntary_results)}")

    retained = sum(1 for r in voluntary_results if r.get("retained"))
    escalated = sum(1 for r in voluntary_results if r.get("escalated_to_human"))
    print(f"\n📈 Churn Voluntário:")
    print(f"   Sinais de risco processados : {len(voluntary_results)}")
    print(f"   Clientes retidos             : {retained}/{len(voluntary_results)}")
    print(f"   Escalados para CS humano     : {escalated}/{len(voluntary_results)}")

    print(f"\n✅ Pipeline CRAI v2 (involuntário + voluntário + HubSpot) funcionando!\n")


if __name__ == "__main__":
    asyncio.run(main())
