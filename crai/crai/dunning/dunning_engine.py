"""crai/dunning/dunning_engine.py — LangGraph + Claude API: dunning multicanal (churn involuntário)."""

from anthropic import AsyncAnthropic
from langgraph.graph import StateGraph, END
from typing import TypedDict

claude = AsyncAnthropic()

# Chaves = códigos de erro do Stripe (mesmo vocabulário de AgentState.failure_cause).
FALLBACK_TEMPLATES = {
    "expired_card":       "Olá! Seu cartão expirou e não conseguimos cobrar R$ {amount:.2f}. Atualize em: {link}",
    "insufficient_funds": "Oi! Tivemos dificuldade ao cobrar R$ {amount:.2f}. Acesse para regularizar: {link}",
    "do_not_honor":       "Olá! Seu banco bloqueou uma cobrança de R$ {amount:.2f}. Acesse: {link}",
    "card_declined":      "Olá! Seu banco recusou a cobrança de R$ {amount:.2f}. Acesse: {link}",
    "generic_decline":    "Olá! Seu banco recusou a cobrança de R$ {amount:.2f}. Acesse: {link}",
    "processing_error":   "Olá! Houve uma falha técnica no pagamento de R$ {amount:.2f}. Detalhes: {link}",
}


class DunningState(TypedDict):
    customer_id: str
    failure_cause: str
    recovery_score: float
    amount: float
    channel: str
    tone: str
    message: str
    sent: bool
    portal_link: str


class DunningEngine:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(DunningState)
        g.add_node("select_channel", self._select_channel)
        g.add_node("define_tone", self._define_tone)
        g.add_node("generate_msg", self._generate_message)
        g.add_node("send_msg", self._send_message)
        g.set_entry_point("select_channel")
        g.add_edge("select_channel", "define_tone")
        g.add_edge("define_tone", "generate_msg")
        g.add_edge("generate_msg", "send_msg")
        g.add_edge("send_msg", END)
        return g.compile()

    async def _select_channel(self, state):
        # O canal ja vem decidido pela politica do Modulo 5; whatsapp e o
        # default para chamadas legadas que nao informam um.
        portal = f"https://pay.crai.ai/{state['customer_id'][:8]}"
        return {**state, "channel": state.get("channel") or "whatsapp", "portal_link": portal}

    async def _define_tone(self, state):
        score = state["recovery_score"]
        tone = "empático" if score >= 0.7 else "amigável" if score >= 0.4 else "direto"
        return {**state, "tone": tone}

    async def _generate_message(self, state):
        prompt = f"""Gere mensagem de {state['channel']} para recuperar pagamento falho.
Causa: {state['failure_cause']} | Valor: R$ {state['amount']:.2f} | Tom: {state['tone']}
Máximo 3 frases. Português brasileiro natural. Incluir link: {state['portal_link']}
Retorne APENAS a mensagem."""
        try:
            response = await claude.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            message = response.content[0].text.strip()
        except Exception as e:
            print(f"[DUNNING] Claude API indisponível ({e}) — usando fallback")
            template = FALLBACK_TEMPLATES.get(state["failure_cause"], FALLBACK_TEMPLATES["processing_error"])
            message = template.format(amount=state["amount"], link=state["portal_link"])
        return {**state, "message": message}

    async def _send_message(self, state):
        print(f"[DUNNING] {state['channel'].upper()} → {state['customer_id']}: {state['message'][:90]}")
        return {**state, "sent": True}

    async def run_campaign(self, customer_id, failure_cause, recovery_score, amount,
                           canal: str = "whatsapp") -> dict:
        initial = DunningState(
            customer_id=customer_id, failure_cause=failure_cause, recovery_score=recovery_score,
            amount=amount, channel=canal, tone="", message="", sent=False, portal_link="",
        )
        result = await self.graph.ainvoke(initial)
        return {"sent": result["sent"], "channel": result["channel"], "message": result["message"]}
