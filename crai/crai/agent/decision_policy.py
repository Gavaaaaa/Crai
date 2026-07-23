"""crai/agent/decision_policy.py — Modulo 5: politica de decisao de cobranca.

Decide a proxima acao para uma fatura em aberto maximizando e-Profit
esperado, e devolve o raciocinio em portugues para o log de auditoria.

    retentativa | e-mail | WhatsApp | Pix/boleto | ligacao CS | nao intervir

Os parametros (crenca de retentativa por causa, taxa de resposta por canal e
limiar de escalonamento humano) sao calibrados em modulo_05_decision_policy/
e carregados de crai/models/decision_policy.json. Sem o arquivo, usa os
mesmos valores declarados aqui — o pipeline nunca quebra por falta dele.

Validacao (5.000 cobrancas simuladas): +62,9% de receita liquida contra a
regra fixa de mercado, 98,0% do teto do oraculo.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
POLICY_PATH = MODELS_DIR / "decision_policy.json"

# ── Priors declarados (fallback sem o arquivo calibrado) ─────────────────
CRENCA_RETRY = {
    "insufficient_funds": 0.75,
    "processing_error":   0.85,
    "generic_decline":    0.25,
    "card_declined":      0.15,
    "do_not_honor":       0.05,
    "expired_card":       0.00,   # cartao vencido nao passa em retentativa
}
CRENCA_RESPOSTA = {"email": 0.25, "whatsapp": 0.45, "pix_boleto": 0.55, "ligacao_cs": 0.75}
CUSTO_ACAO = {"retry": 0.02, "email": 0.02, "whatsapp": 0.05,
              "pix_boleto": 0.50, "ligacao_cs": 15.00}
LIMIAR_MARGINAL_CS = 200.0
CANAIS_AUTO = ["email", "whatsapp", "pix_boleto"]

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


class DecisionPolicy:
    """Escolhe a proxima acao de cobranca maximizando e-Profit esperado."""

    def __init__(self):
        self.crenca_retry = dict(CRENCA_RETRY)
        self.crenca_resposta = dict(CRENCA_RESPOSTA)
        self.custo = dict(CUSTO_ACAO)
        self.limiar_cs = LIMIAR_MARGINAL_CS
        self.is_calibrada = False

    def load(self) -> bool:
        """Carrega os parametros calibrados de crai/models/decision_policy.json."""
        try:
            with open(POLICY_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            self.crenca_retry = cfg["crenca_retry"]
            self.crenca_resposta = cfg["crenca_resposta"]
            self.custo = cfg["custo_acao"]
            self.limiar_cs = float(cfg["limiar_marginal_cs"])
            self.is_calibrada = True
            v = cfg.get("validacao", {})
            print(f"[POLICY] Politica calibrada carregada de {POLICY_PATH.name} "
                  f"(+{v.get('ganho_pct', 0)}% vs regra fixa em "
                  f"{v.get('n_cobrancas', 0)} cobrancas simuladas)")
            return True
        except FileNotFoundError:
            print("[POLICY] decision_policy.json nao encontrado — usando priors declarados")
            return False
        except Exception as e:
            print(f"[POLICY] Erro ao carregar politica: {e}")
            return False

    # ── Decisao ──────────────────────────────────────────────────────────
    def decidir(self, ctx: dict, ja_tentou_retry: bool = False,
                cs_disponivel: bool = True) -> dict:
        causa = ctx.get("causa", "processing_error")
        valor = float(ctx.get("valor", 0.0))
        p = float(ctx.get("p_recovery", 0.5))
        dia = int(ctx.get("payday_previsto", 0)) if causa == "insufficient_funds" else 0

        ep_retry = self.crenca_retry.get(causa, 0.2) * valor - self.custo["retry"]
        autos = {c: p * self.crenca_resposta[c] * valor - self.custo[c] for c in CANAIS_AUTO}
        melhor_auto = max(autos, key=autos.get)
        ep_auto = autos[melhor_auto]
        ep_cs = p * self.crenca_resposta["ligacao_cs"] * valor - self.custo["ligacao_cs"]
        ganho_marginal = ep_cs - ep_auto

        if not ja_tentou_retry and ep_retry >= ep_auto and ep_retry > 0:
            espera = (f" — agendada para o dia {dia}, quando a previsao de liquidez "
                      f"indica saldo" if causa == "insufficient_funds" else "")
            return self._saida("retry", dia, ep_retry, False,
                               f"{CAUSA_PT.get(causa, causa)}: a retentativa tem e-Profit "
                               f"R$ {ep_retry:.2f}, acima do melhor canal de contato "
                               f"(R$ {ep_auto:.2f}){espera}.")

        if cs_disponivel and ganho_marginal > self.limiar_cs:
            return self._saida("ligacao_cs", dia, ep_cs, True,
                               f"Fatura de R$ {valor:.2f} com recuperabilidade {p:.0%}: a "
                               f"ligacao humana rende R$ {ganho_marginal:.2f} a mais que "
                               f"{ROTULO_ACAO[melhor_auto]}, acima do limiar de "
                               f"R$ {self.limiar_cs:.2f} para uso do time de CS.")

        if ep_auto <= 0:
            return self._saida("nao_intervir", dia, ep_auto, False,
                               f"Nenhum canal tem e-Profit positivo (melhor: "
                               f"R$ {ep_auto:.2f}). Intervir destroi valor.")

        motivo_retry = (" Retentar nao resolve esta causa."
                        if self.crenca_retry.get(causa, 0.2) == 0.0 else "")
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
