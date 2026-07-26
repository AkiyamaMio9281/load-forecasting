"""Single-layer LSTM (SPEC section 5, optional).

Architecture, kept deliberately small:

    last 168 hourly loads --> LSTM(1 layer, 64 hidden) --> final hidden state
                                                              |
    target day's exogenous block (24 x 6, flattened) ---------+--> MLP --> 24 outputs

So it is a *direct* 24-output model, like the LightGBM ensemble, and sees exactly
the same information: a week of history ending at the cutoff, plus the calendar and
(perfect-forecast) temperature of the target day.

Two things are worth stating plainly rather than hiding:
  * With ~6k training days and a strong hand-built feature set already available to
    the trees, a net this size is not expected to win. It is included to cover the
    deep-learning comparison honestly, and the result is reported either way.
  * Refitting from scratch in all 52 folds is the expensive part. Epochs are
    deliberately few, and `torch` is an optional dependency -- importing this module
    without it fails loudly rather than silently degrading the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON
from src.models.base import BaseModel

SEQ_LEN = 168  # one week of history feeds the encoder
EXOG_COLS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "temp", "is_holiday")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the lstm model needs pytorch: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        ) from exc
    return torch


class LstmModel(BaseModel):
    """Sequence encoder + exogenous head, refit per fold."""

    name = "lstm"
    max_train_years: float | None = 6.0

    def __init__(
        self,
        seed: int = 42,
        hidden: int = 64,
        epochs: int = 12,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net_ = None

    # --- data plumbing ------------------------------------------------------- #
    def _build_net(self, exog_dim: int):
        torch = _require_torch()
        from torch import nn

        hidden = self.hidden

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden + exog_dim, 128), nn.ReLU(), nn.Linear(128, HORIZON)
                )

            def forward(self, sequence, exog):
                _, (h_n, _) = self.encoder(sequence)
                return self.head(torch.cat([h_n[-1], exog], dim=1))

        return Net()

    def _assemble(
        self, table: pd.DataFrame, days: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack per-day (history sequence, exogenous block, target) tensors."""
        y_by_ts = table["y"]
        sequences, exogs, targets = [], [], []
        for day in days:
            rows = table[table["op_date"] == day]
            if len(rows) != HORIZON:
                continue
            cutoff = rows["cutoff_ts"].iloc[0]
            window = y_by_ts.loc[cutoff - pd.Timedelta(hours=SEQ_LEN - 1) : cutoff]
            if len(window) != SEQ_LEN or window.isna().any():
                continue
            sequences.append(window.to_numpy())
            exogs.append(rows[list(EXOG_COLS)].to_numpy().ravel())
            targets.append(rows["y"].to_numpy())
        return np.asarray(sequences), np.asarray(exogs), np.asarray(targets)

    # --- interface ------------------------------------------------------------ #
    def fit(self, history: pd.DataFrame) -> None:
        torch = _require_torch()
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        history = self._truncate(history)
        usable = history[history["bad_day"] == 0]
        days = sorted({d for d in usable["op_date"] if (usable["op_date"] == d).sum() == HORIZON})
        sequences, exogs, targets = self._assemble(history, days)
        if len(sequences) < 50:
            raise ValueError(f"lstm: only {len(sequences)} usable training days")

        # Standardise on the training fold only -- statistics are part of the model.
        self.y_mean_, self.y_std_ = float(targets.mean()), float(targets.std() + 1e-8)
        self.exog_mean_ = exogs.mean(axis=0)
        self.exog_std_ = exogs.std(axis=0) + 1e-8

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = device
        x_seq = torch.tensor(
            ((sequences - self.y_mean_) / self.y_std_)[..., None], dtype=torch.float32
        )
        x_exog = torch.tensor((exogs - self.exog_mean_) / self.exog_std_, dtype=torch.float32)
        y = torch.tensor((targets - self.y_mean_) / self.y_std_, dtype=torch.float32)

        net = self._build_net(exog_dim=x_exog.shape[1]).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()
        loader = DataLoader(
            TensorDataset(x_seq, x_exog, y), batch_size=self.batch_size, shuffle=True
        )

        net.train()
        for _ in range(self.epochs):
            for seq_batch, exog_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(net(seq_batch.to(device), exog_batch.to(device)), y_batch.to(device))
                loss.backward()
                optimizer.step()

        net.eval()
        self.net_ = net
        self._history_y = history["y"]

    def predict(
        self, horizon_index: pd.DatetimeIndex, future_exog: pd.DataFrame | None
    ) -> np.ndarray:
        torch = _require_torch()
        if self.net_ is None:
            raise RuntimeError("lstm: predict called before fit")
        if future_exog is None:
            raise ValueError("lstm requires future_exog")

        cutoff = horizon_index[0] - pd.Timedelta(hours=1)
        window = self._history_y.loc[cutoff - pd.Timedelta(hours=SEQ_LEN - 1) : cutoff].to_numpy()
        if len(window) != SEQ_LEN:
            raise ValueError(f"lstm: need {SEQ_LEN}h of history, got {len(window)}")

        exog = future_exog[list(EXOG_COLS)].to_numpy().ravel()[None, :]
        x_seq = torch.tensor(
            ((window - self.y_mean_) / self.y_std_)[None, :, None], dtype=torch.float32
        )
        x_exog = torch.tensor((exog - self.exog_mean_) / self.exog_std_, dtype=torch.float32)

        with torch.no_grad():
            scaled = (
                self.net_(x_seq.to(self.device_), x_exog.to(self.device_)).cpu().numpy().ravel()
            )
        return scaled * self.y_std_ + self.y_mean_
