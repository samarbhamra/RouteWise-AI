"""PyTorch LSTM for hourly segment travel-time sequences."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import LSTM_META_PATH, LSTM_MODEL_PATH, LSTM_SEQ_LEN
from src.features.build import TABULAR_FEATURES, TARGET


class TravelLSTM(nn.Module):
    def __init__(self, static_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        h = h[-1]
        return self.head(torch.cat([h, static], dim=1)).squeeze(-1)


def build_sequences(df: pd.DataFrame, seq_len: int = LSTM_SEQ_LEN):
    """Build (seq, static, y) from segment-hour rows using 24h profiles."""
    seqs, statics, ys, keys = [], [], [], []
    static_cols = [c for c in TABULAR_FEATURES if c not in ("hour",)]
    grouped = df.groupby(["u", "v", "k", "dow"])
    for (u, v, k, dow), g in grouped:
        by_hour = g.groupby("hour")[TARGET].mean()
        profile = np.array([float(by_hour.get(h, by_hour.mean())) for h in range(24)], dtype=np.float32)
        for _, row in g.iterrows():
            hour = int(row["hour"])
            idx = [(hour - seq_len + i) % 24 for i in range(seq_len)]
            seqs.append(profile[idx].reshape(seq_len, 1))
            statics.append(row[static_cols].fillna(0).to_numpy(dtype=np.float32))
            ys.append(float(row[TARGET]))
            keys.append((int(u), int(v), int(k), int(row["hour"]), int(dow)))
    return (
        np.stack(seqs),
        np.stack(statics),
        np.array(ys, dtype=np.float32),
        keys,
        static_cols,
    )


class LSTMTravelTime:
    def __init__(self, model=None, static_cols=None, seq_len: int = LSTM_SEQ_LEN):
        self.model = model
        self.static_cols = static_cols
        self.seq_len = seq_len
        self.device = torch.device("cpu")

    def fit(self, train: pd.DataFrame, epochs: int = 8, batch_size: int = 256, lr: float = 1e-3):
        seq, static, y, _, static_cols = build_sequences(train, self.seq_len)
        self.static_cols = static_cols
        self.model = TravelLSTM(static_dim=static.shape[1]).to(self.device)
        ds = TensorDataset(
            torch.tensor(seq),
            torch.tensor(static),
            torch.tensor(y),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        self.model.train()
        for _ in range(epochs):
            for xs, st, yt in loader:
                pred = self.model(xs, st)
                loss = loss_fn(pred, yt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_arrays(self, seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(seq), torch.tensor(static))
        return out.numpy()

    def evaluate(self, df: pd.DataFrame) -> dict:
        seq, static, y, _, _ = build_sequences(df, self.seq_len)
        pred = self.predict_arrays(seq, static)
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
        }

    def predict_row(self, row: pd.Series, hour_profile: np.ndarray) -> float:
        hour = int(row["hour"])
        idx = [(hour - self.seq_len + i) % 24 for i in range(self.seq_len)]
        seq = hour_profile[idx].reshape(1, self.seq_len, 1).astype(np.float32)
        static = row[self.static_cols].fillna(0).to_numpy(dtype=np.float32).reshape(1, -1)
        return float(self.predict_arrays(seq, static)[0])

    def save(self, path: Path | None = None):
        path = path or LSTM_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        joblib.dump(
            {"static_cols": self.static_cols, "seq_len": self.seq_len, "static_dim": len(self.static_cols)},
            LSTM_META_PATH,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "LSTMTravelTime":
        path = path or LSTM_MODEL_PATH
        meta = joblib.load(LSTM_META_PATH)
        inst = cls(seq_len=meta["seq_len"], static_cols=meta["static_cols"])
        model = TravelLSTM(static_dim=meta["static_dim"])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        inst.model = model
        return inst
