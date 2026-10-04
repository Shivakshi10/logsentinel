"""LSTM autoencoder: learns to reconstruct normal sequences of windows.

The model sees the last ``seq_len`` windows, compresses them into a small
hidden state and tries to reconstruct them. Trained only on healthy data, it
reconstructs normal behaviour well and anomalies badly, so the reconstruction
error of the newest window is the anomaly score.

Unlike the Isolation Forest, this model sees features that never varied during
training (after standardisation they explode), so it also catches brand-new
error messages.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Detector

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "The LSTM autoencoder needs PyTorch. Install it with "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        )


if nn is not None:

    class _Network(nn.Module):
        def __init__(self, n_features: int, hidden: int, latent: int):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.decoder = nn.LSTM(latent, hidden, batch_first=True)
            self.output = nn.Linear(hidden, n_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])  # (batch, latent)
            repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
            decoded, _ = self.decoder(repeated)
            return self.output(decoded)


class LSTMAutoencoderDetector(Detector):
    name = "lstm_autoencoder"

    def __init__(
        self,
        seq_len: int = 10,
        hidden: int = 64,
        latent: int = 16,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 1e-3,
        clip: float = 50.0,
        random_state: int = 42,
    ):
        _require_torch()
        self.context = seq_len
        self.seq_len = seq_len
        self.hidden = hidden
        self.latent = latent
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.clip = clip
        self.random_state = random_state
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.network = None
        self.train_losses: list[float] = []

    # -- helpers -----------------------------------------------------------------
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean_) / self.std_, -self.clip, self.clip).astype(np.float32)

    def _sequences(self, Z: np.ndarray) -> np.ndarray:
        """Sliding sequences ending at every row; early rows are left-padded."""
        pad = np.repeat(Z[:1], self.seq_len - 1, axis=0)
        padded = np.vstack([pad, Z])
        idx = np.arange(self.seq_len)[None, :] + np.arange(len(Z))[:, None]
        return padded[idx]

    # -- Detector API --------------------------------------------------------------
    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> LSTMAutoencoderDetector:
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), 0.01)
        sequences = torch.from_numpy(self._sequences(self._standardise(X)))

        self.network = _Network(X.shape[1], self.hidden, self.latent)
        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.network.train()
        self.train_losses = []
        for _ in range(self.epochs):
            order = rng.permutation(len(sequences))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = sequences[order[start : start + self.batch_size]]
                optimiser.zero_grad()
                loss = loss_fn(self.network(batch), batch)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)
            self.train_losses.append(total / len(sequences))
        self.network.eval()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Detector must be fitted before scoring")
        sequences = torch.from_numpy(self._sequences(self._standardise(X)))
        scores = []
        with torch.no_grad():
            for start in range(0, len(sequences), 512):
                batch = sequences[start : start + 512]
                error = (self.network(batch)[:, -1, :] - batch[:, -1, :]) ** 2
                scores.append(error.mean(dim=1).numpy())
        return np.concatenate(scores) if scores else np.zeros(0)

    # -- persistence: store weights, not the live module ---------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["network"] = self.network.state_dict() if self.network is not None else None
        state["n_features"] = len(self.mean_) if self.mean_ is not None else None
        return state

    def __setstate__(self, state: dict) -> None:
        _require_torch()
        weights = state.pop("network")
        n_features = state.pop("n_features")
        self.__dict__.update(state)
        self.network = None
        if weights is not None:
            self.network = _Network(n_features, self.hidden, self.latent)
            self.network.load_state_dict(weights)
            self.network.eval()
