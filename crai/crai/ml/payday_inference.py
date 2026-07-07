"""crai/ml/payday_inference.py — Módulo 3: LSTM + Prophet (Inferência de Liquidez).

Prevê QUANDO o cliente terá saldo para a retentativa: uma LSTM projeta 14 dias
de probabilidade de liquidez a partir dos últimos 30 dias de saldo do cliente,
e um Prophet por perfil (CLT/PJ/freelancer) contribui o prior sazonal do
payday brasileiro (5º dia útil, dias 10/15/20/30). Ensemble 0.6/0.4.

Cold start (modelo não treinado): heurística de dias fixos por perfil.
Produção: artefatos treinados pelo pipeline em modulo_03_payday/.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

try:
    from prophet.serialize import model_from_json
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROPHET_AVAILABLE = False

# ── Diretório de persistência (mesmo padrão dos módulos 1 e 2) ────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

WINDOW = 30
HORIZON = 14
PESO_LSTM = 0.6
LIMIAR = 0.5
PROFILES = ["CLT", "PJ", "freelancer"]

PAYDAY_HEURISTICS = {
    "CLT":        [5, 6, 7, 20, 21],
    "freelancer": [10, 15, 20, 25],
    "PJ":         [5, 10, 15, 20, 25],
    "default":    [5, 20],
}


if TORCH_AVAILABLE:
    class LiquidityLSTM(nn.Module):
        """Arquitetura idêntica à treinada em modulo_03_payday/src/model.py."""

        def __init__(self, input_size: int = 5, hidden_size: int = 64,
                     num_layers: int = 2, horizon: int = HORIZON, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, horizon),
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])


class PaydayInference:
    """Fase 1: heurística por perfil. Fase 2: LSTM + Prophet treinados."""

    def __init__(self):
        self.is_fitted = False
        self.model = None
        self.priors = {}

    # ── Carregamento dos modelos treinados ───────────────────────────────
    def load(self) -> bool:
        """Carrega LSTM + Prophets por perfil de crai/models/."""
        if not TORCH_AVAILABLE or not PROPHET_AVAILABLE:
            print("[PAYDAY] torch/prophet não instalados — usando heurística")
            return False
        try:
            with open(MODELS_DIR / "payday_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            self.model = LiquidityLSTM(hidden_size=meta["hidden_size"])
            self.model.load_state_dict(torch.load(MODELS_DIR / "payday_lstm.pt"))
            self.model.eval()
            for p in PROFILES:
                with open(MODELS_DIR / f"payday_prophet_{p}.json", encoding="utf-8") as f:
                    self.priors[p] = model_from_json(f.read())
            self.is_fitted = True
            print(f"[PAYDAY] LSTM + {len(self.priors)} Prophets carregados de {MODELS_DIR}/")
            return True
        except FileNotFoundError:
            print("[PAYDAY] Modelos não encontrados — usando heurística")
            return False
        except Exception as e:
            print(f"[PAYDAY] Erro ao carregar modelos: {e}")
            return False

    # ── Interface consumida pelo agente (Módulo 5) ───────────────────────
    async def predict_next_window(self, customer_id: str) -> dict:
        if not self.is_fitted:
            history = await self._fetch_history(customer_id)
            profile = self._classify_profile(history)
            return {**self._heuristic_predict(profile), "method": "heuristic"}

        serie = self._liquidity_series(customer_id)
        profile = serie.attrs["profile"]

        X = torch.from_numpy(self._featurize(serie)[None, ...])
        with torch.no_grad():
            probs_lstm = torch.sigmoid(self.model(X)).numpy()[0]

        hoje = pd.Timestamp(datetime.now().date())
        datas_futuras = pd.date_range(hoje + pd.Timedelta(days=1), periods=HORIZON)
        futuro = pd.DataFrame({"ds": datas_futuras})
        prior = np.clip(self.priors[profile].predict(futuro)["yhat"].to_numpy(), 0.0, 1.0)

        probs = PESO_LSTM * probs_lstm + (1 - PESO_LSTM) * prior
        acima = np.where(probs >= LIMIAR)[0]
        idx = int(acima[0]) if len(acima) else int(np.argmax(probs))

        return {
            "timestamp": (datas_futuras[idx] + pd.Timedelta(hours=10)).to_pydatetime(),
            "confidence": round(float(probs[idx]), 4),
            "profile": profile,
            "method": "lstm_prophet",
        }

    # ── Featurização (idêntica a modulo_03_payday/src/features.py) ───────
    @staticmethod
    def _featurize(serie: pd.DataFrame) -> np.ndarray:
        dia = serie["day_of_month"].to_numpy()
        return np.stack([
            serie["has_liquidity"].to_numpy(dtype=np.float32),
            np.clip(serie["balance_norm"].to_numpy(dtype=np.float32), 0, 5),
            np.sin(2 * np.pi * dia / 31).astype(np.float32),
            np.cos(2 * np.pi * dia / 31).astype(np.float32),
            (serie["weekday"].to_numpy() < 5).astype(np.float32),
        ], axis=1)

    def _liquidity_series(self, customer_id: str) -> pd.DataFrame:
        """Últimos 30 dias de saldo do cliente (em produção: gateway/open finance).

        Simulação determinística por customer_id, com as mesmas âncoras de
        pagamento BR usadas no treino (CLT 50%, PJ 30%, freelancer 20%).
        """
        rng = np.random.default_rng(seed=abs(hash(customer_id)) % (2**32))
        profile = rng.choice(PROFILES, p=[0.50, 0.30, 0.20])

        hoje = pd.Timestamp(datetime.now().date())
        datas = pd.date_range(hoje - pd.Timedelta(days=WINDOW - 1), periods=WINDOW)

        entradas = np.zeros(WINDOW)
        if profile == "CLT":
            salario = rng.uniform(2.5, 5.0)
            bday_idx = pd.Series((datas.dayofweek < 5)).groupby(
                [datas.year, datas.month]).cumsum().where(datas.dayofweek < 5, 0)
            entradas[np.asarray(bday_idx) == 5] = salario * 0.6
            entradas[datas.day == int(np.clip(rng.normal(20, 1), 18, 22))] += salario * 0.4
            gasto = rng.uniform(0.08, 0.14)
        elif profile == "PJ":
            receita = rng.uniform(2.0, 6.0)
            for ancora, peso in [(10, 0.4), (15, 0.3), (28, 0.3)]:
                if rng.uniform() > 0.15:
                    entradas[datas.day == ancora + int(rng.integers(0, 4))] += \
                        receita * peso * rng.uniform(0.7, 1.3)
            gasto = rng.uniform(0.10, 0.18)
        else:
            n_pag = int(rng.integers(2, 6))
            entradas[rng.choice(WINDOW, size=n_pag, replace=False)] = \
                rng.exponential(scale=1.2, size=n_pag)
            gasto = rng.uniform(0.10, 0.20)

        saldo = np.zeros(WINDOW)
        atual = rng.uniform(0.2, 1.5)
        for i in range(WINDOW):
            atual = max(0.0, atual + entradas[i] - gasto * rng.uniform(0.5, 1.5))
            saldo[i] = atual

        serie = pd.DataFrame({
            "date": datas, "day_of_month": datas.day, "weekday": datas.dayofweek,
            "balance_norm": np.round(saldo, 4),
            "has_liquidity": (saldo >= 1.0).astype(int),
        })
        serie.attrs["profile"] = str(profile)
        return serie

    # ── Fallback heurístico (cold start) ─────────────────────────────────
    def _classify_profile(self, history: list) -> str:
        amounts = [h["amount"] for h in history if h.get("status") == "succeeded"]
        if len(amounts) < 3:
            return "CLT"
        cv = np.std(amounts) / (np.mean(amounts) + 1e-9)
        if cv < 0.15: return "CLT"
        if cv < 0.40: return "PJ"
        return "freelancer"

    def _heuristic_predict(self, profile: str) -> dict:
        paydays = PAYDAY_HEURISTICS.get(profile, PAYDAY_HEURISTICS["default"])
        today = datetime.now()
        next_date = None
        for day in sorted(paydays):
            candidate = today.replace(day=min(day, 28))
            if candidate > today + timedelta(hours=2):
                next_date = candidate
                break
        if next_date is None:
            next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
            next_date = next_month.replace(day=paydays[0])
        return {
            "timestamp":  next_date,
            "confidence": 0.65 if profile == "CLT" else 0.45,
            "profile":    profile,
        }

    async def _fetch_history(self, customer_id: str) -> list:
        rng = np.random.default_rng(seed=abs(hash(customer_id)) % (2**32))
        base = rng.uniform(500, 5000)
        noise = rng.normal(0, base * 0.05, 30)
        return [
            {"amount": round(float(base + noise[i]), 2), "date": datetime.now() - timedelta(days=i),
             "status": "succeeded" if rng.random() > 0.05 else "failed"}
            for i in range(30)
        ]
