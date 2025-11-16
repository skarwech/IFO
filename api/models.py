"""
Pydantic models for API request/response validation.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class PumpMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    OFF = "off"


class PumpStatus(BaseModel):
    """Individual pump status."""
    id: int = Field(..., description="Pump identifier (1-4)")
    frequency: float = Field(..., description="Current frequency in Hz", ge=0, le=50)
    flow: float = Field(..., description="Current flow in m³/h", ge=0)
    power: float = Field(..., description="Power consumption in kW", ge=0)
    mode: PumpMode = Field(default=PumpMode.AUTO)
    is_running: bool = Field(default=True)


class TunnelMetrics(BaseModel):
    """Current tunnel state metrics."""
    volume: float = Field(..., description="Current volume in m³", ge=0)
    level: float = Field(..., description="Water level in meters", ge=0)
    inflow_rate: float = Field(..., description="Current inflow rate in m³/h", ge=0)
    outflow_rate: float = Field(..., description="Total outflow rate in m³/h", ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)


class SystemStatus(BaseModel):
    """Overall system status."""
    pumps: List[PumpStatus]
    tunnel: TunnelMetrics
    total_power: float = Field(..., description="Total power consumption in kW", ge=0)
    optimization_active: bool = Field(default=False)
    last_optimization: Optional[datetime] = None
    system_mode: str = Field(default="offline", pattern="^(offline|realtime|hybrid)$")


class ForecastData(BaseModel):
    """Forecasted inflow data."""
    model_config = {"protected_namespaces": ()}
    
    timestamps: List[datetime]
    inflow_predictions: List[float] = Field(..., description="Predicted inflow in m³/h")
    confidence_intervals: Optional[List[Dict[str, float]]] = Field(
        None, 
        description="Lower and upper bounds for each prediction"
    )
    model_type: str = Field(default="LSTM", description="Forecast model used")


class OptimizationRequest(BaseModel):
    """Request for optimization run."""
    horizon: int = Field(default=96, description="Optimization horizon in timesteps", ge=1, le=288)
    mode: str = Field(default="hybrid", pattern="^(offline|realtime|hybrid)$")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Custom constraints")


class FileUploadConfig(BaseModel):
    """Configuration for file upload and data processing."""
    skip_rows: Optional[List[int]] = Field(default=None, description="Row indices to skip (0-based)")
    drop_columns: Optional[List[str]] = Field(default=None, description="Column names to drop")
    inflow_column: str = Field(default="Inflow to tunnel F1", description="Column name for inflow data")
    timestamp_column: Optional[str] = Field(default="Time stamp", description="Column name for timestamps")
    use_for: str = Field(default="both", pattern="^(forecast|optimize|both)$", description="What to use data for")


class FileUploadResponse(BaseModel):
    """Response after file upload and processing."""
    status: str
    rows_processed: int
    columns_found: List[str]
    data_preview: List[Dict[str, Any]] = Field(description="First 5 rows of processed data")
    forecast_result: Optional[ForecastData] = None
    optimization_result: Optional["OptimizationResult"] = None
    execution_result: Optional[Dict[str, Any]] = Field(None, description="Execution results (frequencies applied, flow, power)")
    message: str


class OptimizationResult(BaseModel):
    """Optimization results."""
    schedule: List[List[float]] = Field(..., description="Pump frequencies [timesteps x pumps]")
    predicted_power: List[float] = Field(..., description="Power consumption per timestep")
    predicted_volume: List[float] = Field(..., description="Tunnel volume per timestep")
    total_energy: float = Field(..., description="Total energy consumption in kWh", ge=0)
    cost_savings: Optional[float] = Field(None, description="Estimated cost savings vs baseline")
    computation_time: float = Field(..., description="Optimization time in seconds", ge=0)
    status: str = Field(..., description="Optimization status (optimal/feasible/infeasible)")


class HistoricalDataQuery(BaseModel):
    """Query for historical data."""
    start_time: datetime
    end_time: datetime
    metrics: List[str] = Field(
        default=["power", "flow", "volume"],
        description="Metrics to retrieve"
    )
    aggregation: Optional[str] = Field(None, pattern="^(raw|1min|5min|15min|1h)$")


class HistoricalDataResponse(BaseModel):
    """Historical data response."""
    timestamps: List[datetime]
    data: Dict[str, List[float]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KPIMetrics(BaseModel):
    """Key Performance Indicators."""
    total_energy_consumed: float = Field(..., description="Total energy in kWh")
    average_power: float = Field(..., description="Average power in kW")
    peak_power: float = Field(..., description="Peak power in kW")
    energy_cost: float = Field(..., description="Energy cost in currency")
    cost_savings_vs_baseline: float = Field(..., description="Cost savings percentage")
    volume_compliance: float = Field(..., description="Volume constraint compliance %", ge=0, le=100)
    avg_tunnel_level: float = Field(..., description="Average tunnel level in meters")
    period_start: datetime
    period_end: datetime


class LiveUpdate(BaseModel):
    """WebSocket live update message."""
    timestamp: datetime = Field(default_factory=datetime.now)
    pumps: List[PumpStatus]
    tunnel: TunnelMetrics
    total_power: float
    event_type: str = Field(default="status_update", pattern="^(status_update|alert|optimization_complete)$")
    message: Optional[str] = None


class EdgeMetricsPayload(BaseModel):
    """Payload for edge device metrics reporting."""
    model_config = {"protected_namespaces": ()}
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatMessage(BaseModel):
    """Chat message for conversation history."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request for chatbot conversation."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    include_system_status: bool = Field(default=True, description="Include current system status in context")


class ChatResponse(BaseModel):
    """Response from chatbot."""
    response: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """API error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
