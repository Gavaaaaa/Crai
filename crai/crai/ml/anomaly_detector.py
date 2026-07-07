"""crai/ml/anomaly_detector.py — Módulo 2: Autoencoder de anomalias comportamentais.

Autoencoder denso (12→32→16→4→16→32→12) treinado apenas em clientes saudáveis.
O erro de reconstrução funciona como score de anomalia: clientes cujo
comportamento se desviou do padrão saudável são reconstruídos com erro alto.

Cold start (modelo não treinado): heurística de z-score sobre o evento Stripe.
Produção: autoencoder treinado pelo pipeline em modulo_02_autoencoder/.
"""

import json
from pathlib import Path

import joblib
import numpy as np

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

# ── Diretório de persistência (mesmo padrão do failure_classifier) ───────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

BEHAVIORAL_FEATURES = [
    "tenure_days", "mrr_brl", "seats",
    "logins_7d", "logins_30d", "feature_adoption",
    "avg_session_min", "api_calls_7d", "days_since_last_login",
    "tickets_30d", "failed_pay_90d", "nps_last",
]


if TORCH_AVAILABLE:
    class BehaviorAutoencoder(nn.Module):
        """Arquitetura idêntica à treinada em modulo_02_autoencoder/src/model.py."""

        def __init__(self, input_dim: int = 12, bottleneck: int = 4, dropout: float = 0.1):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


class AnomalyDetector:
    """Cold start: heurística de z-score. Produção: autoencoder comportamental."""

    THRESHOLDS = {"CLT": 0.15, "PJ": 0.35, "freelancer": 0.55, "default": 0.30}

    def __init__(self):
        self.is_fitted = False
        self.model = None
        self.scaler = None
        self.threshold = None
        self.features = BEHAVIORAL_FEATURES

    # ── Carregamento do modelo treinado ──────────────────────────────────
    def load(self) -> bool:
        """Carrega autoencoder, scaler e threshold de crai/models/."""
        if not TORCH_AVAILABLE:
            print("[ANOMALY] PyTorch não instalado — usando heurística")
            return False
        try:
            with open(MODELS_DIR / "autoencoder_meta.json", encoding="utf-8") as f:
                meta = json.load(f)

            self.features = meta["features"]
            self.threshold = float(meta["threshold"])
            self.model = BehaviorAutoencoder(
                input_dim=meta["input_dim"], bottleneck=meta["bottleneck"]
            )
            self.model.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt"))
            self.model.eval()
            self.scaler = joblib.load(MODELS_DIR / "autoencoder_scaler.pkl")

            self.is_fitted = True
            print(f"[ANOMALY] Autoencoder carregado de {MODELS_DIR}/ (threshold={self.threshold:.4f})")
            return True
        except FileNotFoundError:
            print("[ANOMALY] Autoencoder não encontrado — usando heurística")
            return False
        except Exception as e:
            print(f"[ANOMALY] Erro ao carregar autoencoder: {e}")
            return False

    # ── Interface consumida pelo agente (Módulo 5) ───────────────────────
    async def check(self, customer_id: str, event: dict) -> dict:
        if not self.is_fitted:
            return self._heuristic_check(event)

        behavior = self._behavioral_snapshot(customer_id)
        X = self.scaler.transform(
            np.array([[behavior[f] for f in self.features]], dtype=np.float32)
        ).astype(np.float32)

        with torch.no_grad():
            tensor = torch.from_numpy(X)
            recon = self.model(tensor).numpy()

        erro_por_feature = (recon[0] - X[0]) ** 2
        error = float(erro_por_feature.mean())

        ranking = sorted(
            zip(self.features, erro_por_feature.tolist()),
            key=lambda t: t[1], reverse=True,
        )
        return {
            "is_anomaly": error > self.threshold,
            "error": round(error, 4),
            "threshold": self.threshold,
            "method": "autoencoder",
            "top_features": [
                {"feature": nome, "contribuicao": round(contrib, 4)}
                for nome, contrib in ranking[:3]
            ],
        }

    def _behavioral_snapshot(self, customer_id: str) -> dict:
        """Snapshot comportamental do cliente (em produção viria do Segment/DB).

        Determinístico por customer_id: ~15% dos clientes exibem o padrão
        degradado (queda de uso, fricção alta) usado no treino como anomalia.
        """
        rng = np.random.default_rng(seed=abs(hash(customer_id)) % (2**32))
        degradado = rng.uniform() < 0.15

        seats = int(np.clip(rng.poisson(lam=15), 1, 200))
        if degradado:
            logins_30d = int(max(0, rng.normal(loc=seats * 5, scale=seats * 2)))
            logins_7d = int(logins_30d * rng.uniform(0.05, 0.15))
            adoption = round(float(rng.beta(2, 5)), 3)
            session = round(float(max(0.5, rng.normal(6, 3))), 1)
            api_calls = int(max(0, rng.lognormal(4.5, 1.0)))
            last_login = int(np.clip(rng.exponential(scale=9), 0, 30))
            tickets = int(rng.poisson(4.5))
            failed_pay = int(rng.binomial(3, 0.35))
            nps = round(float(np.clip(rng.normal(5.5, 2.0), 0, 10)), 1)
        else:
            logins_30d = int(max(1, rng.normal(loc=seats * 18, scale=seats * 3)))
            logins_7d = int(logins_30d * rng.uniform(0.22, 0.30))
            adoption = round(float(rng.beta(5, 2)), 3)
            session = round(float(max(2, rng.normal(22, 6))), 1)
            api_calls = int(max(10, rng.lognormal(7.0, 0.8)))
            last_login = int(np.clip(rng.exponential(scale=1.5), 0, 30))
            tickets = int(rng.poisson(1.2))
            failed_pay = int(rng.binomial(3, 0.05))
            nps = round(float(np.clip(rng.normal(8.2, 1.3), 0, 10)), 1)

        return {
            "tenure_days": int(np.clip(rng.gamma(2.5, 180), 30, 2000)),
            "mrr_brl": round(float(np.clip(rng.lognormal(8.5, 0.7), 500, 50_000)), 2),
            "seats": seats,
            "logins_7d": logins_7d,
            "logins_30d": logins_30d,
            "feature_adoption": adoption,
            "avg_session_min": session,
            "api_calls_7d": api_calls,
            "days_since_last_login": last_login,
            "tickets_30d": tickets,
            "failed_pay_90d": failed_pay,
            "nps_last": nps,
        }

    # ── Fallback heurístico (cold start) ─────────────────────────────────
    def _heuristic_check(self, event: dict) -> dict:
        features  = self._extract_features(event)
        profile   = event.get("profile_type", "default")
        threshold = self.THRESHOLDS.get(profile, self.THRESHOLDS["default"])
        error     = self._simple_outlier_score(features)
        return {
            "is_anomaly": error > threshold,
            "error":      round(error, 4),
            "threshold":  threshold,
            "method":     "heuristic",
            "top_features": [],
        }

    def _simple_outlier_score(self, features: np.ndarray) -> float:
        if len(features) == 0:
            return 0.1
        mean = np.mean(features)
        std  = np.std(features) + 1e-9
        z    = np.abs((features - mean) / std)
        return float(np.clip(np.mean(z) / 5.0, 0, 1))

    def _extract_features(self, event: dict) -> np.ndarray:
        charge = event.get("data", {}).get("object", {})
        return np.array([
            charge.get("amount", 0) / 100,
            len(charge.get("failure_code", "") or ""),
            int(bool(charge.get("failure_message"))),
            charge.get("attempt_count", 1),
        ], dtype=float)
