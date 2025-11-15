"""
LSTM-based forecasting for inflow prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[StandardScaler] = None
    ):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences of shape (num_samples, seq_len, num_features)
            y: Target sequences of shape (num_samples, forecast_len)
            transform: Optional scaler for normalization
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """
    LSTM model for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            output_size: Length of forecast horizon
            dropout: Dropout rate
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        forecast = self.fc(last_output)
        
        return forecast


class InflowForecaster:
    """
    Forecasting agent for wastewater inflow.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize forecaster.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.forecast_config = config['forecasting']['lstm']
        
        self.input_hours = self.forecast_config['input_hours']
        self.output_hours = self.forecast_config['output_hours']
        self.hidden_size = self.forecast_config['hidden_size']
        self.num_layers = self.forecast_config['num_layers']
        self.dropout = self.forecast_config['dropout']
        self.learning_rate = self.forecast_config['learning_rate']
        self.batch_size = self.forecast_config['batch_size']
        self.epochs = self.forecast_config['epochs']
        
        self.timestep_minutes = config['simulation']['timestep_minutes']
        
        # Calculate sequence lengths
        self.input_steps = (self.input_hours * 60) // self.timestep_minutes
        self.output_steps = (self.output_hours * 60) // self.timestep_minutes
        
        # Model and scalers
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'F1') -> np.ndarray:
        """
        Prepare features from DataFrame.
        
        Args:
            df: Input DataFrame with time series
            target_col: Target column name
            
        Returns:
            Feature array
        """
        features = []
        
        # Target variable
        features.append(df[target_col].values.reshape(-1, 1))
        
        # Time features (if available)
        if 'hour_of_day' in df.columns:
            features.append(df['hour_of_day'].values.reshape(-1, 1) / 24.0)
        if 'day_of_week' in df.columns:
            features.append(df['day_of_week'].values.reshape(-1, 1) / 7.0)
        if 'is_weekend' in df.columns:
            features.append(df['is_weekend'].values.reshape(-1, 1))
        
        # Cyclical features
        if 'hour_sin' in df.columns:
            features.append(df['hour_sin'].values.reshape(-1, 1))
            features.append(df['hour_cos'].values.reshape(-1, 1))
        if 'day_sin' in df.columns:
            features.append(df['day_sin'].values.reshape(-1, 1))
            features.append(df['day_cos'].values.reshape(-1, 1))
        
        # Rolling features
        rolling_cols = [col for col in df.columns if 'rolling' in col and target_col in col]
        for col in rolling_cols:
            features.append(df[col].values.reshape(-1, 1))
        
        # Concatenate all features
        X = np.concatenate(features, axis=1)
        
        return X
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for training.
        
        Args:
            X: Feature array of shape (num_timesteps, num_features)
            y: Target array of shape (num_timesteps,)
            
        Returns:
            X_seq, y_seq arrays
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.input_steps - self.output_steps + 1):
            X_seq.append(X[i:i + self.input_steps])
            y_seq.append(y[i + self.input_steps:i + self.input_steps + self.output_steps])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, df: pd.DataFrame, target_col: str = 'F1'):
        """
        Train the LSTM model.
        
        Args:
            df: Training DataFrame
            target_col: Target column to forecast
        """
        logger.info("Preparing training data...")
        
        # Prepare features
        X = self.prepare_features(df, target_col)
        y = df[target_col].values
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        logger.info(f"Created {len(X_seq)} sequences")
        logger.info(f"Input shape: {X_seq.shape}, Output shape: {y_seq.shape}")
        
        # Normalize
        num_samples, seq_len, num_features = X_seq.shape
        X_seq_reshaped = X_seq.reshape(-1, num_features)
        X_seq_scaled = self.scaler_X.fit_transform(X_seq_reshaped)
        X_seq_scaled = X_seq_scaled.reshape(num_samples, seq_len, num_features)
        
        y_seq_scaled = self.scaler_y.fit_transform(y_seq)
        
        # Split train/validation
        split_idx = int(0.8 * len(X_seq_scaled))
        X_train, X_val = X_seq_scaled[:split_idx], X_seq_scaled[split_idx:]
        y_train, y_val = y_seq_scaled[:split_idx], y_seq_scaled[split_idx:]
        
        # Create datasets and loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Initialize model
        self.model = LSTMForecaster(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_steps,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        logger.info("Training LSTM model...")
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    
    def fit(self, inflow_series: np.ndarray):
        """
        Fit the forecaster to historical inflow data (simplified interface).
        
        Args:
            inflow_series: Historical inflow values (1D array)
        """
        # Store statistics for simple persistence/statistical forecasting
        self.inflow_mean = np.mean(inflow_series)
        self.inflow_std = np.std(inflow_series)
        self.inflow_min = np.min(inflow_series)
        self.inflow_max = np.max(inflow_series)
        
        # Store recent history for persistence model
        history_len = min(len(inflow_series), self.input_steps)
        self.inflow_history = inflow_series[-history_len:]
        
        logger.info(
            f"Inflow forecaster fitted: mean={self.inflow_mean:.1f}, "
            f"std={self.inflow_std:.1f}, n={len(inflow_series)}"
        )
    
    def forecast(
        self,
        current_volume_or_df,
        horizon_hours_or_target: Optional[int] = None,
        last_n_hours: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate forecast (supports multiple interfaces).
        
        Args:
            current_volume_or_df: Either current volume (float) or DataFrame
            horizon_hours_or_target: Either horizon hours (int) or target column (str)
            last_n_hours: Use last N hours (for DataFrame interface)
            
        Returns:
            Forecast array
        """
        # Detect interface
        if isinstance(current_volume_or_df, (int, float)):
            # Simple interface: forecast(volume, horizon_hours)
            current_volume = current_volume_or_df
            horizon_hours = horizon_hours_or_target or self.output_hours
            
            # Use simple persistence/statistical model
            n_steps = (horizon_hours * 60) // self.timestep_minutes
            
            if hasattr(self, 'inflow_history') and len(self.inflow_history) > 0:
                # Persistence model with slight noise
                last_value = self.inflow_history[-1]
                forecast = np.ones(n_steps) * last_value
                # Add small random variation
                noise = np.random.normal(0, self.inflow_std * 0.1, n_steps)
                forecast += noise
                forecast = np.clip(forecast, self.inflow_min, self.inflow_max)
            else:
                # Fallback to mean
                forecast = np.ones(n_steps) * getattr(self, 'inflow_mean', 1600.0)
            
            return forecast
        
        else:
            # DataFrame interface: forecast(df, target_col, last_n_hours)
            df = current_volume_or_df
            target_col = horizon_hours_or_target or 'F1'
            
            if self.model is None:
                # No trained model - use simple statistical forecast
                logger.warning("No trained LSTM model, using statistical forecast")
                
                if target_col in df.columns:
                    recent_values = df[target_col].values[-24:]
                    mean_val = np.mean(recent_values)
                else:
                    mean_val = getattr(self, 'inflow_mean', 1600.0)
                
                return np.ones(self.output_steps) * mean_val
            
            if last_n_hours is None:
                last_n_hours = self.input_hours
            
            # Get last N steps
            n_steps = (last_n_hours * 60) // self.timestep_minutes
            df_recent = df.iloc[-n_steps:]
            
            # Prepare features
            X = self.prepare_features(df_recent, target_col)
            
            # Ensure we have enough data
            if len(X) < self.input_steps:
                logger.warning(f"Insufficient data for forecast. Need {self.input_steps}, got {len(X)}")
                # Pad with last value
                padding = np.repeat(X[-1:], self.input_steps - len(X), axis=0)
                X = np.vstack([padding, X])
            
            # Take last input_steps
            X = X[-self.input_steps:]
            
            # Normalize
            X_scaled = self.scaler_X.transform(X)
            X_scaled = X_scaled.reshape(1, self.input_steps, -1)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                y_pred_scaled = self.model(X_tensor).cpu().numpy()
            
            # Inverse transform
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            
            return y_pred.flatten()
    
    def detect_rain(self, forecast: np.ndarray, threshold: float = 2000) -> bool:
        """
        Detect if rain is forecasted.
        
        Args:
            forecast: Forecast array (in m³/15min)
            threshold: Threshold for rain detection
            
        Returns:
            True if rain detected
        """
        return np.any(forecast > threshold)
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': {
                'input_steps': self.input_steps,
                'output_steps': self.output_steps,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore config
        config = checkpoint['config']
        self.input_steps = config['input_steps']
        self.output_steps = config['output_steps']
        
        # Restore scalers
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        # Rebuild model
        # Need to infer input size from scaler
        input_size = self.scaler_X.n_features_in_
        
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=self.output_steps,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {path}")
