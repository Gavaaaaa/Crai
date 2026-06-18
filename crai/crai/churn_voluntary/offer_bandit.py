"""
crai/churn_voluntary/offer_bandit.py
Multi-Armed Bandit (Epsilon-Greedy) — aprende qual oferta converte melhor
por perfil de cliente, sem precisar de regras programadas manualmente.

Cada "braço" do bandit é uma oferta. O algoritmo testa todas no início
(exploração) e progressivamente passa a usar a que mais converte
(explotação), por perfil (CLT / PJ / freelancer).
"""

import random
from collections import defaultdict

OFFERS = ["desconto_10", "desconto_20", "pausa_1_mes", "consulta_cs"]


class OfferBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit.

    epsilon = probabilidade de explorar (testar oferta aleatória)
    Com o tempo, baixamos epsilon para favorecer a oferta vencedora.
    """

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon
        # stats[profile][offer] = {"shown": int, "accepted": int}
        self.stats = defaultdict(lambda: {o: {"shown": 0, "accepted": 0} for o in OFFERS})

        # Priors realistas para cold start (baseados em benchmarks de mercado)
        self._seed_priors()

    def _seed_priors(self):
        priors = {
            "CLT":        {"desconto_10": (20, 9),  "desconto_20": (20, 14), "pausa_1_mes": (20, 8),  "consulta_cs": (10, 7)},
            "PJ":         {"desconto_10": (15, 5),  "desconto_20": (15, 8),  "pausa_1_mes": (20, 13), "consulta_cs": (15, 11)},
            "freelancer": {"desconto_10": (20, 9),  "desconto_20": (20, 9),  "pausa_1_mes": (15, 6),  "consulta_cs": (10, 6)},
        }
        for profile, offers in priors.items():
            for offer, (shown, accepted) in offers.items():
                self.stats[profile][offer] = {"shown": shown, "accepted": accepted}

    def choose_offer(self, profile: str, risk_score: float) -> str:
        """
        Escolhe a oferta. Risco muito alto (>=0.90) pula direto para
        consulta_cs — intervenção humana é a única aposta segura.
        """
        if risk_score >= 0.90:
            return "consulta_cs"

        if random.random() < self.epsilon:
            return random.choice(OFFERS)  # exploração

        # explotação: escolhe a de maior taxa de conversão histórica
        profile_stats = self.stats[profile]
        best_offer = max(
            profile_stats,
            key=lambda o: profile_stats[o]["accepted"] / max(profile_stats[o]["shown"], 1)
        )
        return best_offer

    def record_outcome(self, profile: str, offer: str, accepted: bool):
        """Atualiza as estatísticas após saber se o cliente aceitou."""
        self.stats[profile][offer]["shown"] += 1
        if accepted:
            self.stats[profile][offer]["accepted"] += 1

    def conversion_rates(self, profile: str) -> dict:
        """Retorna taxa de conversão atual de cada oferta para um perfil."""
        return {
            o: round(s["accepted"] / max(s["shown"], 1), 3)
            for o, s in self.stats[profile].items()
        }
