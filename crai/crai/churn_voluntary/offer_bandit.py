"""crai/churn_voluntary/offer_bandit.py — Módulo 4: Thompson Sampling (MAB).

Cada par (perfil, oferta) mantém um posterior Beta(α, β) sobre a taxa de
aceite. A cada decisão o bandit AMOSTRA uma taxa de cada posterior e escolhe
a oferta que maximiza o e-Profit com a taxa amostrada:

    escolha = argmax_o  p_amostrado(o) × LTV_retido − custo(o)

A incerteza faz a exploração sozinha: braços pouco testados têm posteriores
largos e às vezes amostram alto; braços ruins saem de cena. Não há epsilon
para calibrar. O aprendizado é contínuo: cada aceite/recusa real atualiza o
posterior e é persistido em crai/models/bandit_state.json.

Warm start: posteriores da simulação de 6.000 rodadas do modulo_04_offer_bandit/.
Cold start (sem arquivo): priors de benchmarks de mercado.
"""

import json
from pathlib import Path

import numpy as np

# ── Diretório de persistência (mesmo padrão dos módulos 1-3) ─────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
STATE_PATH = MODELS_DIR / "bandit_state.json"

OFFERS = ["desconto_10", "desconto_20", "pausa_1_mes", "consulta_cs", "pix_boleto_flash"]
PROFILES = ["CLT", "PJ", "freelancer"]

MESES_LTV_RETIDO = 6          # valor retido em caso de aceite (consistente com o Módulo 1)
MRR_TIPICO = {"CLT": 300.0, "PJ": 550.0, "freelancer": 220.0, "default": 350.0}

# Priors de cold start (benchmarks de mercado), em pseudo-observações (aceites, recusas)
SEED_PRIORS = {
    "CLT":        {"desconto_10": (9, 11),  "desconto_20": (14, 6), "pausa_1_mes": (8, 12),
                   "consulta_cs": (7, 3),   "pix_boleto_flash": (2, 8)},
    "PJ":         {"desconto_10": (5, 10),  "desconto_20": (8, 7),  "pausa_1_mes": (13, 7),
                   "consulta_cs": (11, 4),  "pix_boleto_flash": (3, 7)},
    "freelancer": {"desconto_10": (9, 11),  "desconto_20": (9, 11), "pausa_1_mes": (6, 9),
                   "consulta_cs": (6, 4),   "pix_boleto_flash": (5, 5)},
}


def offer_cost(offer: str, mrr: float) -> float:
    """Custo da intervenção em R$ (descontos custam % do MRR por 3 meses)."""
    return {
        "desconto_10": 0.10 * 3 * mrr,
        "desconto_20": 0.20 * 3 * mrr,
        "pausa_1_mes": 1.0 * mrr,
        "consulta_cs": 250.0,
        "pix_boleto_flash": 2.0,
    }[offer]


class OfferBandit:
    """Thompson Sampling (Beta-Bernoulli) otimizando e-Profit por perfil."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.is_fitted = False
        self.state = {
            p: {o: {"alpha": 1.0 + a, "beta": 1.0 + b} for o, (a, b) in SEED_PRIORS[p].items()}
            for p in PROFILES
        }

    # ── Carregamento do warm start ───────────────────────────────────────
    def load(self) -> bool:
        """Carrega posteriores aprendidos de crai/models/bandit_state.json."""
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                self.state = json.load(f)
            self.is_fitted = True
            n_obs = sum(s["alpha"] + s["beta"] - 2 for p in self.state.values() for s in p.values())
            print(f"[BANDIT] Posteriores carregados de {STATE_PATH} ({n_obs:.0f} observações)")
            return True
        except FileNotFoundError:
            print("[BANDIT] Estado não encontrado — usando priors de benchmark (cold start)")
            return False
        except Exception as e:
            print(f"[BANDIT] Erro ao carregar estado: {e}")
            return False

    # ── Interface consumida pelo agente (Módulo 5) ───────────────────────
    def choose_offer(self, profile: str, risk_score: float, mrr: float | None = None) -> str:
        """Risco crítico (>= 0.90) escala direto para humano; senão, Thompson."""
        if risk_score >= 0.90:
            return "consulta_cs"

        if mrr is None:
            mrr = MRR_TIPICO.get(profile, MRR_TIPICO["default"])
        ltv_retido = MESES_LTV_RETIDO * mrr

        perfil = self.state.get(profile, self.state["CLT"])
        amostras = {o: self.rng.beta(s["alpha"], s["beta"]) for o, s in perfil.items()}
        return max(amostras, key=lambda o: amostras[o] * ltv_retido - offer_cost(o, mrr))

    def record_outcome(self, profile: str, offer: str, accepted: bool):
        """Atualiza o posterior após saber se o cliente aceitou e persiste."""
        s = self.state.setdefault(profile, {o: {"alpha": 1.0, "beta": 1.0} for o in OFFERS})
        s = s.setdefault(offer, {"alpha": 1.0, "beta": 1.0})
        if accepted:
            s["alpha"] += 1.0
        else:
            s["beta"] += 1.0
        self._persist()

    def conversion_rates(self, profile: str) -> dict:
        """Média do posterior por oferta para um perfil."""
        perfil = self.state.get(profile, self.state["CLT"])
        return {
            o: round(s["alpha"] / (s["alpha"] + s["beta"]), 3)
            for o, s in perfil.items()
        }

    def _persist(self):
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"[BANDIT] Falha ao persistir estado: {e}")
