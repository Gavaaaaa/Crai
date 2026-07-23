"""Interface publica do Modulo 5 — consumida pelo agente LangGraph.

Enquanto `policies.py` existe para a simulacao comparativa, esta classe e o
que o pipeline do CRAI chama em producao. Alem da acao, devolve o RACIOCINIO
em portugues: qual alternativa foi considerada, qual venceu e por que — o
log de decisao que o TCC promete para auditoria.

    from src.decision_policy import DecisionPolicy

    pol = DecisionPolicy()
    pol.decidir({"causa": "expired_card", "valor": 149.0, "ltv": 1800.0,
                 "p_recovery": 0.9, "payday_previsto": 0})
    # {"acao": "whatsapp", "dia_offset": 0, "eprofit_esperado": 60.28,
    #  "escalar_humano": False,
    #  "motivo": "Cartao expirado nao passa em retentativa (...)"}
"""
from __future__ import annotations

from src.environment import CUSTO_ACAO
from src.policies import CRENCA_RESPOSTA, CRENCA_RETRY, LIMIAR_MARGINAL_CS

ROTULO_ACAO = {
    "retry": "retentativa automatica",
    "email": "e-mail",
    "whatsapp": "WhatsApp",
    "pix_boleto": "link Pix/boleto",
    "ligacao_cs": "ligacao do time de CS",
}
CAUSA_PT = {
    "insufficient_funds": "Saldo insuficiente",
    "expired_card": "Cartao expirado",
    "card_declined": "Cartao recusado",
    "processing_error": "Erro tecnico do gateway",
    "do_not_honor": "Banco nao honrou",
    "generic_decline": "Recusa generica",
}
CANAIS_AUTO = ["email", "whatsapp", "pix_boleto"]


class DecisionPolicy:
    """Escolhe a proxima acao de cobranca maximizando e-Profit esperado."""

    def __init__(self, crenca_retry: dict | None = None,
                 crenca_resposta: dict | None = None,
                 limiar_cs: float = LIMIAR_MARGINAL_CS) -> None:
        self.crenca_retry = dict(crenca_retry or CRENCA_RETRY)
        self.crenca_resposta = dict(crenca_resposta or CRENCA_RESPOSTA)
        self.limiar_cs = limiar_cs

    def decidir(self, ctx: dict, ja_tentou_retry: bool = False,
                cs_disponivel: bool = True) -> dict:
        causa = ctx["causa"]
        valor = float(ctx["valor"])
        p = float(ctx.get("p_recovery", 0.5))

        dia = int(ctx.get("payday_previsto", 0)) if causa == "insufficient_funds" else 0

        ep_retry = self.crenca_retry.get(causa, 0.2) * valor - CUSTO_ACAO["retry"]
        autos = {c: p * self.crenca_resposta[c] * valor - CUSTO_ACAO[c] for c in CANAIS_AUTO}
        melhor_auto = max(autos, key=autos.get)
        ep_auto = autos[melhor_auto]
        ep_cs = p * self.crenca_resposta["ligacao_cs"] * valor - CUSTO_ACAO["ligacao_cs"]
        ganho_marginal = ep_cs - ep_auto

        # 1) Retentativa, quando a causa e mecanicamente retentavel.
        #    O gate de e-Profit vale aqui tambem: retentar de graca nao existe.
        if not ja_tentou_retry and ep_retry >= ep_auto and ep_retry > 0:
            espera = (f" — agendada para o dia {dia}, quando a previsao de liquidez "
                      f"indica saldo" if causa == "insufficient_funds" else "")
            return self._saida("retry", dia, ep_retry, False,
                               f"{CAUSA_PT.get(causa, causa)}: a retentativa tem e-Profit "
                               f"R$ {ep_retry:.2f}, acima do melhor canal de contato "
                               f"(R$ {ep_auto:.2f}){espera}.")

        # 2) Escalar para humano so quando o ganho marginal paga o recurso escasso
        if cs_disponivel and ganho_marginal > self.limiar_cs:
            return self._saida("ligacao_cs", dia, ep_cs, True,
                               f"Fatura de R$ {valor:.2f} com recuperabilidade {p:.0%}: a "
                               f"ligacao humana rende R$ {ganho_marginal:.2f} a mais que "
                               f"{ROTULO_ACAO[melhor_auto]}, acima do limiar de "
                               f"R$ {self.limiar_cs:.2f} para uso do time de CS.")

        # 3) Melhor canal automatico — ou nao intervir
        if ep_auto <= 0:
            return self._saida("nao_intervir", dia, ep_auto, False,
                               f"Nenhum canal tem e-Profit positivo (melhor: "
                               f"R$ {ep_auto:.2f}). Intervir destroi valor.")

        motivo_retry = ""
        if self.crenca_retry.get(causa, 0.2) == 0.0:
            motivo_retry = " Retentar nao resolve esta causa."
        return self._saida(melhor_auto, dia, ep_auto, False,
                           f"{CAUSA_PT.get(causa, causa)}:{motivo_retry} "
                           f"{ROTULO_ACAO[melhor_auto]} maximiza o e-Profit "
                           f"(R$ {ep_auto:.2f}); a ligacao renderia so R$ "
                           f"{ganho_marginal:.2f} a mais, abaixo do limiar.")

    @staticmethod
    def _saida(acao, dia, ep, escalar, motivo) -> dict:
        return {"acao": acao, "dia_offset": int(dia),
                "eprofit_esperado": round(float(ep), 2),
                "escalar_humano": bool(escalar), "motivo": motivo.strip()}


if __name__ == "__main__":
    pol = DecisionPolicy()
    casos = [
        ("Cartao expirado, fatura media", {"causa": "expired_card", "valor": 149.0,
                                           "ltv": 1800.0, "p_recovery": 0.90}),
        ("Saldo insuficiente, paga dia 5", {"causa": "insufficient_funds", "valor": 299.90,
                                            "ltv": 3600.0, "p_recovery": 0.85,
                                            "payday_previsto": 5}),
        ("Erro tecnico", {"causa": "processing_error", "valor": 99.90,
                          "ltv": 1200.0, "p_recovery": 0.95}),
        ("Fatura alta, banco recusou", {"causa": "do_not_honor", "valor": 4800.0,
                                        "ltv": 40000.0, "p_recovery": 0.35}),
    ]
    for nome, ctx in casos:
        d = pol.decidir(ctx)
        print(f"\n{nome}")
        print(f"  -> {d['acao']} (dia +{d['dia_offset']}) | e-Profit R$ {d['eprofit_esperado']:.2f}")
        print(f"  [RACIOCINIO] {d['motivo']}")
