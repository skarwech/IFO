import math
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 15-min interval horizon mapping
DEFAULT_STEP_MINUTES = 15
logger = logging.getLogger(__name__)

class LSTMInflowModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.proj(last)

class InflowForecaster:
    """LSTM forecaster with simple rolling-window training.

    Uses 15-min intervals. Trains to predict next-step inflow (m3/15min).
    For multi-step horizon, performs recursive forecasting.
    """
    def __init__(self, lookback_steps: int = 32, hidden_size: int = 32, num_layers: int = 2, device: str = None):
        self.lookback = lookback_steps
        self.model = LSTMInflowModel(1, hidden_size, num_layers)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        self.trained = False

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.scaler_mean) / (self.scaler_std + 1e-8)

    def _inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.scaler_std + 1e-8) + self.scaler_mean

    def fit(self, series: pd.Series, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64) -> None:
        values = series.dropna().astype(float).values
        if len(values) < self.lookback + 10:
            # Not enough data: mark trained but will fallback to mean persistence
            self.scaler_mean = float(np.mean(values))
            self.scaler_std = float(np.std(values) + 1e-6)
            self.trained = True
            return
        self.scaler_mean = float(np.mean(values))
        self.scaler_std = float(np.std(values) + 1e-6)
        scaled = self._scale(values)
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback:i])
            y.append(scaled[i])
        X = np.array(X)
        y = np.array(y)
        tensor_X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        tensor_y = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred.squeeze(-1), yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.size(0)
                count += xb.size(0)
            avg_loss = total_loss / count if count > 0 else 0.0
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        self.trained = True

    def forecast(self, recent_series: pd.Series, horizon_steps: int) -> pd.Series:
        values = recent_series.dropna().astype(float).values
        if len(values) == 0:
            return pd.Series([0.0] * horizon_steps)
        if not self.trained:
            # Fallback simple rule: mean + last deviation damped
            base = np.mean(values)
            last = values[-1]
            forecasts = []
            for k in range(horizon_steps):
                alpha = math.exp(-0.1 * k)
                forecasts.append(base + alpha * (last - base))
            return pd.Series(forecasts)
        self.model.eval()
        window = values[-self.lookback:] if len(values) >= self.lookback else values
        scaled_window = self._scale(window)
        preds = []
        current = scaled_window.copy()
        for _ in range(horizon_steps):
            inp = torch.tensor(current[-self.lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                p = self.model(inp).cpu().numpy()[0, 0]
            preds.append(p)
            current = np.append(current, p)
        forecasts = self._inverse(np.array(preds))
        return pd.Series(forecasts)

    @staticmethod
    def horizon_from_hours(hours: float) -> int:
        return int(round(hours * 60 / DEFAULT_STEP_MINUTES))
