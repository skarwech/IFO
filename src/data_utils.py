"""
Data utilities for loading and processing historical data and volume tables.
Supports both CSV and Excel formats for production deployment.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VolumeCalculator:
    """
    Convert between water level and volume using interpolation.
    Handles volume table from Excel/CSV with different formula types.
    """
    
    def __init__(self, volume_table_path: str, sheet_name: str = 'Taul1'):
        """
        Initialize with volume-level lookup table.
        
        Args:
            volume_table_path: Path to Excel/CSV file with volume table
            sheet_name: Sheet name (for Excel) or ignored for CSV
        """
        self.table_path = volume_table_path
        self.sheet_name = sheet_name
        
        # Load volume table
        self._load_table()
        
        # Create interpolators
        self._setup_interpolators()
        
        logger.info(f"VolumeCalculator initialized from {volume_table_path}")
    
    def _load_table(self):
        """Load volume table from file."""
        path = Path(self.table_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Volume table not found: {self.table_path}")
        
        # Load based on file extension
        if path.suffix.lower() in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.table_path, sheet_name=self.sheet_name)
        elif path.suffix.lower() == '.csv':
            self.df = pd.read_csv(self.table_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Standardize column names (handle various formats)
        col_mapping = {}
        for col in self.df.columns:
            col_lower = col.lower().strip()
            if 'level' in col_lower and 'l1' in col_lower:
                col_mapping[col] = 'Level_L1_m'
            elif 'volume' in col_lower and 'v' in col_lower:
                col_mapping[col] = 'Volume_V_m3'
            elif 'formula' in col_lower:
                col_mapping[col] = 'Formula_type'
        
        self.df = self.df.rename(columns=col_mapping)
        
        # Ensure required columns exist
        if 'Level_L1_m' not in self.df.columns or 'Volume_V_m3' not in self.df.columns:
            raise ValueError("Volume table must have level and volume columns")
        
        # Remove duplicates and sort
        self.df = self.df.drop_duplicates(subset=['Level_L1_m'])
        self.df = self.df.sort_values('Level_L1_m').reset_index(drop=True)
        
        logger.info(f"Loaded volume table: {len(self.df)} rows")
    
    def _setup_interpolators(self):
        """Create interpolation functions."""
        levels = self.df['Level_L1_m'].values
        volumes = self.df['Volume_V_m3'].values
        
        # Volume from level (linear interpolation)
        self.level_to_volume = interp1d(
            levels,
            volumes,
            kind='linear',
            bounds_error=False,
            fill_value=(volumes[0], volumes[-1])
        )
        
        # Level from volume (inverse)
        self.volume_to_level = interp1d(
            volumes,
            levels,
            kind='linear',
            bounds_error=False,
            fill_value=(levels[0], levels[-1])
        )
    
    def get_volume(self, level_m: float) -> float:
        """Convert level to volume."""
        return float(self.level_to_volume(level_m))
    
    def get_level(self, volume_m3: float) -> float:
        """Convert volume to level."""
        return float(self.volume_to_level(volume_m3))
    
    def get_min_volume(self) -> float:
        """Get minimum volume."""
        return float(self.df['Volume_V_m3'].min())
    
    def get_max_volume(self) -> float:
        """Get maximum volume."""
        return float(self.df['Volume_V_m3'].max())
    
    def get_min_level(self) -> float:
        """Get minimum level."""
        return float(self.df['Level_L1_m'].min())
    
    def get_max_level(self) -> float:
        """Get maximum level."""
        return float(self.df['Level_L1_m'].max())


class DataLoader:
    """
    Load and preprocess historical data from Excel/CSV files.
    """
    
    @staticmethod
    def load_historical_data(
        file_path: str,
        sheet_name: str = 'Taul1'
    ) -> pd.DataFrame:
        """
        Load historical data from Excel or CSV file.
        
        Args:
            file_path: Path to data file
            sheet_name: Sheet name for Excel files
            
        Returns:
            DataFrame with standardized columns
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load based on extension
        if path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Standardize column names
        df = DataLoader._standardize_columns(df)
        
        # Convert timestamp if needed
        df = DataLoader._process_timestamp(df)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to consistent format."""
        col_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Timestamp
            if 'time' in col_lower and 'stamp' in col_lower:
                col_mapping[col] = 'timestamp'
            
            # Water level
            elif 'water' in col_lower and 'level' in col_lower and 'l1' in col_lower:
                col_mapping[col] = 'L1'
            elif col_lower == 'l1':
                col_mapping[col] = 'L1'
            
            # Volume
            elif 'volume' in col_lower and ('tunnel' in col_lower or 'v' in col_lower):
                col_mapping[col] = 'V'
            elif col_lower == 'v':
                col_mapping[col] = 'V'
            
            # Flows
            elif 'sum' in col_lower and 'pump' in col_lower and 'f2' in col_lower:
                col_mapping[col] = 'F2'
            elif 'inflow' in col_lower and 'f1' in col_lower:
                col_mapping[col] = 'F1'
            
            # Pump flows (1.1-1.4, 2.1-2.4)
            elif 'pump flow' in col_lower:
                for i in range(1, 3):  # Pump groups 1 and 2
                    for j in range(1, 5):  # Pumps 1-4
                        if f'{i}.{j}' in col_lower:
                            col_mapping[col] = f'pump_flow_{i}_{j}'
            
            # Pump power
            elif 'pump power' in col_lower or 'power intake' in col_lower:
                for i in range(1, 3):
                    for j in range(1, 5):
                        if f'{i}.{j}' in col_lower:
                            col_mapping[col] = f'pump_power_{i}_{j}'
            
            # Pump frequency
            elif 'pump frequency' in col_lower or 'frequency' in col_lower:
                for i in range(1, 3):
                    for j in range(1, 5):
                        if f'{i}.{j}' in col_lower:
                            col_mapping[col] = f'pump_freq_{i}_{j}'
            
            # Electricity prices
            elif 'electricity price 1' in col_lower or 'price 1' in col_lower:
                col_mapping[col] = 'price_high'
            elif 'electricity price 2' in col_lower or 'price 2' in col_lower:
                col_mapping[col] = 'price_normal'
        
        df = df.rename(columns=col_mapping)
        return df
    
    @staticmethod
    def _process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Process timestamp column to datetime."""
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp column found after standardization")
        
        # Check if timestamp is Excel serial date (fractional days since 1900)
        if df['timestamp'].dtype in [np.float64, np.int64]:
            # Assume Excel date format
            # Excel serial date: days since 1900-01-01 (with 1900 leap year bug)
            excel_epoch = pd.Timestamp('1899-12-30')  # Adjust for Excel bug
            df['timestamp'] = df['timestamp'].apply(
                lambda x: excel_epoch + pd.Timedelta(days=x)
            )
        else:
            # Try parsing as datetime string
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df


def create_sample_data(
    duration_days: int = 14,
    interval_minutes: int = 15,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create sample data for testing (mimics Hackathon_HSY_data.xlsx structure).
    
    Args:
        duration_days: Number of days of data
        interval_minutes: Time interval in minutes
        output_path: Optional path to save CSV/Excel
        
    Returns:
        DataFrame with sample data
    """
    n_steps = int(duration_days * 24 * 60 / interval_minutes)
    
    # Generate timestamps
    start_time = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start=start_time, periods=n_steps, freq=f'{interval_minutes}T')
    
    # Generate synthetic data with daily patterns
    hours = np.arange(n_steps) * (interval_minutes / 60)
    
    # Daily pattern for inflow (peak during day)
    daily_pattern = 1500 + 200 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    inflow_f1 = daily_pattern + np.random.normal(0, 50, n_steps)
    inflow_f1 = np.clip(inflow_f1, 1400, 1800)
    
    # Simulate level dynamics
    level_l1 = np.zeros(n_steps)
    volume_v = np.zeros(n_steps)
    level_l1[0] = 2.0
    
    for i in range(1, n_steps):
        # Simple dynamics: volume changes based on inflow
        if i > 0:
            delta_v = (inflow_f1[i] - 1600) * 0.5  # Simplified
            volume_v[i] = volume_v[i-1] + delta_v
            volume_v[i] = np.clip(volume_v[i], 5000, 11000)
            # Approximate level from volume
            level_l1[i] = 1.5 + (volume_v[i] - 5000) / 3000
    
    # Pumped flow (respond to level)
    flow_f2 = 5000 + 1000 * (level_l1 - 2.0)
    flow_f2 = np.clip(flow_f2, 4000, 7500)
    
    # Electricity prices (high during day, normal at night)
    price_high = 0.3 + 0.02 * np.sin(2 * np.pi * hours / 24)
    price_normal = price_high * 0.95
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'L1': level_l1,
        'V': volume_v + 5400,  # Offset to match typical range
        'F2': flow_f2,
        'F1': inflow_f1,
        'price_high': price_high,
        'price_normal': price_normal
    })
    
    # Add pump data (simplified - 2-3 pumps active)
    for i in range(1, 3):
        for j in range(1, 5):
            active = np.random.random(n_steps) > 0.6
            df[f'pump_flow_{i}_{j}'] = active * (1500 + np.random.normal(0, 200, n_steps))
            df[f'pump_power_{i}_{j}'] = df[f'pump_flow_{i}_{j}'] * 0.15  # Approx
            df[f'pump_freq_{i}_{j}'] = active * (47.5 + np.random.random(n_steps) * 2)
    
    logger.info(f"Created sample data: {len(df)} rows, {duration_days} days")
    
    # Save if path provided
    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == '.csv':
            df.to_csv(output_path, index=False)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False, sheet_name='Taul1')
        logger.info(f"Saved sample data to {output_path}")
    
    return df
