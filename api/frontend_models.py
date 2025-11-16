"""
Pydantic models for frontend integration.
Matches data structures expected by Aquaoptai React frontend.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# Dashboard Models
# =============================================================================

class DashboardCard(BaseModel):
    """Dashboard metric card."""
    title: str
    value: str
    subtitle: Optional[str] = None
    max: Optional[str] = None
    icon: str  # Icon name
    color: str  # Color theme
    trend: Optional[float] = None  # Percentage change
    percentage: Optional[float] = None  # Current percentage
    savings: Optional[str] = None


class PumpItem(BaseModel):
    """Pump item for dashboard."""
    id: str
    name: str
    status: str  # active, standby, maintenance
    freq: float
    power: float


class Alert(BaseModel):
    """System alert."""
    type: str  # success, warning, error, info
    message: str
    time: str
    icon: str
    unread: bool = False


class FlowDataPoint(BaseModel):
    """Flow data time series point."""
    time: str  # e.g., "0:00"
    F1: float  # Inflow
    F2: float  # Outflow


class PriceDataPoint(BaseModel):
    """Energy price data point."""
    time: str
    price: float  # EUR/MWh


class DashboardData(BaseModel):
    """Complete dashboard data."""
    cards: List[DashboardCard]
    pumps: List[PumpItem]
    alerts: List[Alert]
    flow_data: List[FlowDataPoint]
    price_data: List[PriceDataPoint]
    current_time: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Forecast Agent Models
# =============================================================================

class ForecastDataPoint(BaseModel):
    """Forecast data point with confidence intervals."""
    hour: int
    predicted: float
    upper: float
    lower: float
    rain: bool = False


class PriceForecastPoint(BaseModel):
    """Energy price forecast point."""
    hour: int
    price: float


class ForecastAgentData(BaseModel):
    """Forecast agent complete data."""
    horizon: int  # 12, 24, 36, 48 hours
    forecast_data: List[ForecastDataPoint]
    price_data: List[PriceForecastPoint]
    peak_inflow: float
    avg_price: float
    forecast_accuracy: float  # Percentage


# =============================================================================
# Planner Agent Models
# =============================================================================

class PumpSchedulePoint(BaseModel):
    """Pump schedule data point (Gantt-style)."""
    hour: str  # e.g., "0:00"
    small1: float  # Frequency or 0 if off
    small3: float
    large1: float
    large2: float
    price: float  # Energy price at this hour


class CostComparison(BaseModel):
    """Cost comparison for different strategies."""
    strategy: str  # Historical, Constant Flow, AI Optimized
    cost: float  # EUR
    savings: float  # Percentage


class PlannerAgentData(BaseModel):
    """Planner agent complete data."""
    schedule_data: List[PumpSchedulePoint]
    cost_comparison: List[CostComparison]
    plan_status: str  # idle, optimizing, ready
    price_scenario: str  # normal, high, low


# =============================================================================
# Executor Agent Models
# =============================================================================

class ExecutionDataPoint(BaseModel):
    """Execution trajectory data point."""
    time: int  # Hour
    level: float  # Tunnel level in meters
    target: float  # Target level


class PumpControl(BaseModel):
    """Pump control state."""
    id: str
    name: str
    active: bool
    freq: float
    target: float


class ExecutorAgentData(BaseModel):
    """Executor agent complete data."""
    is_executing: bool
    current_time: int  # Current hour in 24h cycle
    tunnel_level: float
    execution_data: List[ExecutionDataPoint]
    pumps: List[PumpControl]


# =============================================================================
# Supervisor Agent Models
# =============================================================================

class AgentMetric(BaseModel):
    """Aggregated metric from agents."""
    metric: str
    value: str
    min: str
    max: str
    avg: str
    unit: str


class Communication(BaseModel):
    """Inter-agent communication log."""
    from_: str = Field(..., alias="from")
    to: str
    message: str
    time: str
    type: str  # warning, success, info, error

    class Config:
        populate_by_name = True


class Constraint(BaseModel):
    """Operational constraint status."""
    name: str
    status: bool
    value: str


class SupervisorAgentData(BaseModel):
    """Supervisor agent complete data."""
    aggregated_metrics: List[AgentMetric]
    communications: List[Communication]
    constraints: List[Constraint]


# =============================================================================
# Simulations Models
# =============================================================================

class SimulationComparisonPoint(BaseModel):
    """Simulation cost comparison point."""
    hour: int
    historical: float
    aiOptimized: float
    savings: float


class SimulationLevelPoint(BaseModel):
    """Tunnel level comparison point."""
    hour: int
    historical: float
    aiOptimized: float


class SimulationScenario(BaseModel):
    """Simulation scenario metadata."""
    id: str
    name: str
    date: str
    savings: float


class SimulationData(BaseModel):
    """Simulation complete data."""
    scenarios: List[SimulationScenario]
    current_scenario: str
    comparison_data: List[SimulationComparisonPoint]
    level_data: List[SimulationLevelPoint]
    is_playing: bool
    current_step: int


# =============================================================================
# Reports Models
# =============================================================================

class DailySavingsPoint(BaseModel):
    """Daily savings data point."""
    day: str
    historical: float
    optimized: float
    savings: float


class CostBreakdownItem(BaseModel):
    """Cost breakdown by category."""
    name: str
    value: float
    color: str


class SavingsTrendPoint(BaseModel):
    """Savings trend over time."""
    day: int
    savings: float


class KPICard(BaseModel):
    """KPI metric card."""
    label: str
    value: str
    change: str
    icon: str
    color: str


class JudgingMetric(BaseModel):
    """Junction 2025 judging criteria metric."""
    criterion: str
    score: str
    detail: str


class ReportsData(BaseModel):
    """Reports complete data."""
    time_range: str  # week, month, year
    daily_savings: List[DailySavingsPoint]
    cost_breakdown: List[CostBreakdownItem]
    savings_trend: List[SavingsTrendPoint]
    kpis: List[KPICard]
    judging_metrics: List[JudgingMetric]


# =============================================================================
# System Overview Models
# =============================================================================

class DataFlow(BaseModel):
    """Data flow between components."""
    from_: str = Field(..., alias="from")
    to: str
    data: str
    color: str

    class Config:
        populate_by_name = True


class SystemOverviewData(BaseModel):
    """System overview complete data."""
    agents: List[Dict[str, Any]]
    data_flows: List[DataFlow]
    system_health: float


# =============================================================================
# Settings Models
# =============================================================================

class OPCUAConfig(BaseModel):
    """OPC UA configuration."""
    server_url: str
    namespace: str
    connected: bool


class DataSourceStatus(BaseModel):
    """Data source connection status."""
    name: str
    description: str
    active: bool


class SettingsData(BaseModel):
    """Settings complete data."""
    dark_mode: bool
    opcua: OPCUAConfig
    data_sources: List[DataSourceStatus]
    notifications_enabled: bool


# =============================================================================
# Pump Details Models
# =============================================================================

class PumpCurvePoint(BaseModel):
    """Pump curve data point (H-Q)."""
    flow: float
    head47: Optional[float] = None
    head48: Optional[float] = None
    head49: Optional[float] = None
    head50: Optional[float] = None


class EfficiencyCurvePoint(BaseModel):
    """Efficiency curve data point."""
    h: float
    eff47: Optional[float] = None
    eff48: Optional[float] = None
    eff49: Optional[float] = None
    eff50: Optional[float] = None


class PumpDetailsData(BaseModel):
    """Pump details complete data."""
    selected_pump: str  # small, large
    selected_curve: str  # hq, efficiency, power, npsh
    hq_curve_data: List[PumpCurvePoint]
    efficiency_curve_data: List[EfficiencyCurvePoint]


# =============================================================================
# Notification Models
# =============================================================================

class Notification(BaseModel):
    """System notification."""
    id: str
    type: str  # warning, success, info, error
    title: str
    message: str
    time: str
    icon: str
    unread: bool = True


class NotificationPanel(BaseModel):
    """Notification panel data."""
    notifications: List[Notification]
    unread_count: int


# =============================================================================
# Chat Assistant Models (already defined in models.py)
# =============================================================================

# Using existing ChatMessage, ChatRequest, ChatResponse


# =============================================================================
# Live WebSocket Update Models
# =============================================================================

class LiveSystemUpdate(BaseModel):
    """Real-time system update via WebSocket."""
    type: str = "system_update"
    timestamp: datetime = Field(default_factory=datetime.now)
    dashboard: Optional[DashboardData] = None
    forecast: Optional[ForecastAgentData] = None
    planner: Optional[PlannerAgentData] = None
    executor: Optional[ExecutorAgentData] = None
    supervisor: Optional[SupervisorAgentData] = None
    alerts: List[Alert] = []
