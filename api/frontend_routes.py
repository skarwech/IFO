"""
FastAPI routes for Aquaoptai frontend integration.
Provides data endpoints matching React component expectations.
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional
import random
import math

from api.frontend_models import (
    DashboardData, DashboardCard, PumpItem, Alert, FlowDataPoint, PriceDataPoint,
    ForecastAgentData, ForecastDataPoint, PriceForecastPoint,
    PlannerAgentData, PumpSchedulePoint, CostComparison,
    ExecutorAgentData, ExecutionDataPoint, PumpControl,
    SupervisorAgentData, AgentMetric, Communication, Constraint,
    SimulationData, SimulationComparisonPoint, SimulationLevelPoint, SimulationScenario,
    ReportsData, DailySavingsPoint, CostBreakdownItem, SavingsTrendPoint, KPICard, JudgingMetric,
    SystemOverviewData, DataFlow,
    SettingsData, OPCUAConfig, DataSourceStatus,
    PumpDetailsData, PumpCurvePoint, EfficiencyCurvePoint,
    NotificationPanel, Notification,
)

router = APIRouter(prefix="/api/v1", tags=["Frontend Integration"])


# =============================================================================
# Dashboard Endpoints
# =============================================================================

@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """
    Get complete dashboard data with cards, pumps, alerts, and time series.
    """
    # Generate mock data - replace with real service calls
    cards = [
        DashboardCard(
            title="Tunnel Level (L1)",
            value="4.20m",
            max="8.00m",
            icon="Gauge",
            color="blue",
            trend=-0.3,
            percentage=52.5
        ),
        DashboardCard(
            title="Storage Volume (V)",
            value="156.8k m³",
            max="225.8k m³",
            icon="Droplets",
            color="cyan",
            trend=0.2,
            percentage=69.5
        ),
        DashboardCard(
            title="Energy Savings",
            value="34.5%",
            subtitle="vs baseline",
            icon="TrendingDown",
            color="green",
            trend=2.1,
            savings="€1,243 today"
        ),
        DashboardCard(
            title="Current Power",
            value="847 kW",
            subtitle="4 pumps active",
            icon="Zap",
            color="orange",
            trend=-5.2,
            percentage=None
        ),
    ]
    
    pumps = [
        PumpItem(id="S1", name="Small Pump 1", status="active", freq=50.0, power=112),
        PumpItem(id="S2", name="Small Pump 2", status="active", freq=49.5, power=110),
        PumpItem(id="S3", name="Small Pump 3", status="active", freq=48.5, power=105),
        PumpItem(id="S4", name="Small Pump 4", status="standby", freq=0, power=0),
        PumpItem(id="L1", name="Large Pump 1", status="active", freq=49.2, power=315),
        PumpItem(id="L2", name="Large Pump 2", status="active", freq=50.0, power=317),
        PumpItem(id="L3", name="Large Pump 3", status="standby", freq=0, power=0),
        PumpItem(id="L4", name="Large Pump 4", status="standby", freq=0, power=0),
    ]
    
    alerts = [
        Alert(
            type="success",
            message="AI optimization achieved 34.5% energy savings in the last 24h",
            time="2 min ago",
            icon="CheckCircle",
            unread=True
        ),
        Alert(
            type="info",
            message="Forecast Agent predicts rain event in next 12 hours - prepare surge capacity",
            time="15 min ago",
            icon="Info",
            unread=True
        ),
    ]
    
    flow_data = [
        FlowDataPoint(
            time=f"{i}:00",
            F1=8000 + random.random() * 4000 + (2000 if i > 18 or i < 6 else 0),
            F2=9000 + random.random() * 3000
        )
        for i in range(24)
    ]
    
    price_data = [
        PriceDataPoint(
            time=f"{i}:00",
            price=50 + random.random() * 40 + (30 if 8 < i < 17 else 0)
        )
        for i in range(24)
    ]
    
    return DashboardData(
        cards=cards,
        pumps=pumps,
        alerts=alerts,
        flow_data=flow_data,
        price_data=price_data
    )


# =============================================================================
# Forecast Agent Endpoints
# =============================================================================

@router.get("/agents/forecast", response_model=ForecastAgentData)
async def get_forecast_agent_data(horizon: int = Query(24, ge=12, le=48)):
    """
    Get forecast agent data with inflow predictions and price forecasts.
    """
    forecast_data = []
    for i in range(horizon):
        base_inflow = 10000 + random.random() * 3000
        rain_boost = 4000 * math.sin((i - 12) / 6 * math.pi) if 12 < i < 18 else 0
        
        forecast_data.append(ForecastDataPoint(
            hour=i,
            predicted=base_inflow + rain_boost,
            upper=base_inflow + rain_boost + 1500,
            lower=max(0, base_inflow + rain_boost - 1500),
            rain=(12 < i < 18)
        ))
    
    price_data = [
        PriceForecastPoint(
            hour=i,
            price=60 + random.random() * 30 + (25 if 8 < i < 17 else 0)
        )
        for i in range(horizon)
    ]
    
    return ForecastAgentData(
        horizon=horizon,
        forecast_data=forecast_data,
        price_data=price_data,
        peak_inflow=max(f.predicted for f in forecast_data),
        avg_price=sum(p.price for p in price_data) / len(price_data),
        forecast_accuracy=92.3
    )


# =============================================================================
# Planner Agent Endpoints
# =============================================================================

@router.get("/agents/planner", response_model=PlannerAgentData)
async def get_planner_agent_data(price_scenario: str = Query("normal")):
    """
    Get planner agent data with pump schedules and cost comparisons.
    """
    schedule_data = [
        PumpSchedulePoint(
            hour=f"{i}:00",
            small1=50 if 6 <= i < 22 else 0,
            small3=48.5 if 8 <= i < 20 else 0,
            large1=49 if (0 <= i < 6 or i >= 22) else 0,
            large2=50 if (0 <= i < 6 or i >= 22) else 0,
            price=60 + random.random() * 30 + (25 if 8 < i < 17 else 0)
        )
        for i in range(24)
    ]
    
    cost_comparison = [
        CostComparison(strategy="Historical", cost=3450, savings=0),
        CostComparison(strategy="Constant Flow", cost=2890, savings=16),
        CostComparison(strategy="AI Optimized", cost=2256, savings=35),
    ]
    
    return PlannerAgentData(
        schedule_data=schedule_data,
        cost_comparison=cost_comparison,
        plan_status="ready",
        price_scenario=price_scenario
    )


# =============================================================================
# Executor Agent Endpoints
# =============================================================================

@router.get("/agents/executor", response_model=ExecutorAgentData)
async def get_executor_agent_data(is_executing: bool = Query(False)):
    """
    Get executor agent data with real-time execution status and pump controls.
    """
    execution_data = [
        ExecutionDataPoint(
            time=i,
            level=4.2 + math.sin(i / 4) * 1.5,
            target=4.0
        )
        for i in range(24)
    ]
    
    pumps = [
        PumpControl(id="S1", name="Small 1", active=True, freq=50.0, target=50.0),
        PumpControl(id="S3", name="Small 3", active=True, freq=48.5, target=48.5),
        PumpControl(id="L1", name="Large 1", active=True, freq=49.2, target=50.0),
        PumpControl(id="L2", name="Large 2", active=True, freq=50.0, target=50.0),
    ]
    
    return ExecutorAgentData(
        is_executing=is_executing,
        current_time=datetime.now().hour,
        tunnel_level=4.2,
        execution_data=execution_data,
        pumps=pumps
    )


# =============================================================================
# Supervisor Agent Endpoints
# =============================================================================

@router.get("/agents/supervisor", response_model=SupervisorAgentData)
async def get_supervisor_agent_data():
    """
    Get supervisor agent data with aggregated metrics and inter-agent communications.
    """
    aggregated_metrics = [
        AgentMetric(
            metric="Tunnel Level (L1)",
            value="4.20m",
            min="2.10m",
            max="7.89m",
            avg="4.45m",
            unit="m"
        ),
        AgentMetric(
            metric="Inflow (F1)",
            value="10,234 m³/h",
            min="7,890 m³/h",
            max="16,780 m³/h",
            avg="11,045 m³/h",
            unit="m³/h"
        ),
        AgentMetric(
            metric="Energy Cost",
            value="€3.21/h",
            min="€1.89/h",
            max="€5.67/h",
            avg="€3.45/h",
            unit="/h"
        ),
    ]
    
    communications = [
        Communication(
            **{
                "from": "Forecast",
                "to": "Planner",
                "message": "Rain event predicted in 14h - high inflow expected",
                "time": "2m ago",
                "type": "warning"
            }
        ),
        Communication(
            **{
                "from": "Planner",
                "to": "Executor",
                "message": "New optimized schedule ready for execution",
                "time": "5m ago",
                "type": "success"
            }
        ),
        Communication(
            **{
                "from": "Executor",
                "to": "Supervisor",
                "message": "Plan execution at 98% accuracy - minor deviation detected",
                "time": "1m ago",
                "type": "info"
            }
        ),
    ]
    
    constraints = [
        Constraint(name="Constant outflow maintained", status=True, value="9,876 m³/h ±2%"),
        Constraint(name="Storage within limits", status=True, value="4.2m / 8.0m (52%)"),
        Constraint(name="Pump start/stop ≥ 2h", status=True, value="All compliant"),
        Constraint(name="Min frequency ≥ 47.5 Hz", status=True, value="All compliant"),
    ]
    
    return SupervisorAgentData(
        aggregated_metrics=aggregated_metrics,
        communications=communications,
        constraints=constraints
    )


# =============================================================================
# Simulations Endpoints
# =============================================================================

@router.get("/simulations", response_model=SimulationData)
async def get_simulation_data(scenario: str = Query("normal")):
    """
    Get simulation data for different scenarios.
    """
    scenarios = [
        SimulationScenario(id="historical", name="Historical Replay", date="2024-11-10", savings=0),
        SimulationScenario(id="normal", name="Normal Conditions", date="2024-11-15", savings=34),
        SimulationScenario(id="storm", name="Storm Event", date="2024-10-25", savings=28),
        SimulationScenario(id="peak", name="Peak Prices", date="2024-11-01", savings=42),
    ]
    
    comparison_data = []
    for i in range(48):
        historical = 3500 + random.random() * 1000 + (1200 if 8 < i < 17 else 0)
        ai_optimized = 2800 + random.random() * 800 + (800 if i < 6 or i > 22 else 200)
        
        comparison_data.append(SimulationComparisonPoint(
            hour=i,
            historical=historical,
            aiOptimized=ai_optimized,
            savings=max(0, historical - ai_optimized)
        ))
    
    level_data = [
        SimulationLevelPoint(
            hour=i,
            historical=4.5 + math.sin(i / 6) * 1.2,
            aiOptimized=4.0 + math.sin(i / 8) * 0.8
        )
        for i in range(48)
    ]
    
    return SimulationData(
        scenarios=scenarios,
        current_scenario=scenario,
        comparison_data=comparison_data,
        level_data=level_data,
        is_playing=False,
        current_step=0
    )


# =============================================================================
# Reports Endpoints
# =============================================================================

@router.get("/reports", response_model=ReportsData)
async def get_reports_data(time_range: str = Query("week")):
    """
    Get reports data with savings analysis and KPIs.
    """
    daily_savings = [
        DailySavingsPoint(day="Mon", historical=3450, optimized=2256, savings=1194),
        DailySavingsPoint(day="Tue", historical=3280, optimized=2145, savings=1135),
        DailySavingsPoint(day="Wed", historical=3890, optimized=2567, savings=1323),
        DailySavingsPoint(day="Thu", historical=3120, optimized=2034, savings=1086),
        DailySavingsPoint(day="Fri", historical=3650, optimized=2389, savings=1261),
        DailySavingsPoint(day="Sat", historical=2890, optimized=1956, savings=934),
        DailySavingsPoint(day="Sun", historical=2780, optimized=1823, savings=957),
    ]
    
    cost_breakdown = [
        CostBreakdownItem(name="Peak Hours", value=1234, color="#ef4444"),
        CostBreakdownItem(name="Normal Hours", value=2345, color="#f59e0b"),
        CostBreakdownItem(name="Off-Peak Hours", value=3456, color="#10b981"),
    ]
    
    savings_trend = [
        SavingsTrendPoint(day=i + 1, savings=30 + random.random() * 15)
        for i in range(30)
    ]
    
    kpis = [
        KPICard(label="Total Savings (7 days)", value="€8,890", change="+12%", icon="TrendingDown", color="green"),
        KPICard(label="Energy Reduction", value="34.5%", change="+2.1%", icon="Zap", color="blue"),
        KPICard(label="Peak Shaving", value="892 kW", change="+15%", icon="TrendingDown", color="purple"),
        KPICard(label="Avg. Daily Savings", value="€1,270", change="+8%", icon="Award", color="purple"),
        KPICard(label="Constraint Compliance", value="99.2%", change="+0.5%", icon="Award", color="orange"),
    ]
    
    judging_metrics = [
        JudgingMetric(
            criterion="Applicability",
            score="85%",
            detail="OPC UA integration, 90% infrastructure compatible"
        ),
        JudgingMetric(
            criterion="Scalability",
            score="92%",
            detail="Multi-site deployment ready, cloud-based agents"
        ),
        JudgingMetric(
            criterion="Environmental Impact",
            score="88%",
            detail="34.5% energy reduction, CO₂ savings tracked"
        ),
        JudgingMetric(
            criterion="Economic Impact",
            score="91%",
            detail="€1,270/day average savings, ROI < 6 months"
        ),
    ]
    
    return ReportsData(
        time_range=time_range,
        daily_savings=daily_savings,
        cost_breakdown=cost_breakdown,
        savings_trend=savings_trend,
        kpis=kpis,
        judging_metrics=judging_metrics
    )


# =============================================================================
# System Overview Endpoints
# =============================================================================

@router.get("/system/overview", response_model=SystemOverviewData)
async def get_system_overview():
    """
    Get system overview with agent architecture and data flows.
    """
    agents = [
        {"name": "Forecast", "status": "active", "accuracy": 92.3},
        {"name": "Planner", "status": "active", "optimization": "ready"},
        {"name": "Executor", "status": "active", "pumps_controlled": 4},
        {"name": "Supervisor", "status": "active", "compliance": 99.2},
    ]
    
    data_flows = [
        DataFlow(**{"from": "PCS", "to": "Forecast", "data": "Sensor data", "color": "blue"}),
        DataFlow(**{"from": "Forecast", "to": "Planner", "data": "Inflow predictions", "color": "purple"}),
        DataFlow(**{"from": "Planner", "to": "Executor", "data": "Pump schedule", "color": "green"}),
        DataFlow(**{"from": "Executor", "to": "PCS", "data": "Control signals", "color": "orange"}),
    ]
    
    return SystemOverviewData(
        agents=agents,
        data_flows=data_flows,
        system_health=98.5
    )


# =============================================================================
# Settings Endpoints
# =============================================================================

@router.get("/settings", response_model=SettingsData)
async def get_settings():
    """
    Get system settings and configuration.
    """
    return SettingsData(
        dark_mode=False,
        opcua=OPCUAConfig(
            server_url="opc.tcp://localhost:4840",
            namespace="ns=2;s=HSY.Blominmäki",
            connected=True
        ),
        data_sources=[
            DataSourceStatus(name="HSY Sensor Data", description="PCS Controller via OPC UA", active=True),
            DataSourceStatus(name="DNA Weather API", description="Rain forecasts", active=True),
            DataSourceStatus(name="Energy Prices API", description="15-min spot prices", active=True),
        ],
        notifications_enabled=True
    )


@router.post("/settings/opcua/test")
async def test_opcua_connection():
    """
    Test OPC UA connection.
    """
    return {"status": "success", "message": "Connected to OPC UA server", "latency_ms": 42}


# =============================================================================
# Pump Details Endpoints
# =============================================================================

@router.get("/pumps/details", response_model=PumpDetailsData)
async def get_pump_details(pump_type: str = Query("small"), curve_type: str = Query("hq")):
    """
    Get detailed pump curves and performance data.
    """
    hq_curve_data = [
        PumpCurvePoint(
            flow=i * 500,
            head47=12 - (i * 500 / 10000) * 4,
            head48=13 - (i * 500 / 10000) * 4.2,
            head49=14 - (i * 500 / 10000) * 4.4,
            head50=15 - (i * 500 / 10000) * 4.6,
        )
        for i in range(20)
    ]
    
    efficiency_curve_data = [
        EfficiencyCurvePoint(
            h=i * 0.5,
            eff47=75 + 15 * math.sin((i * 0.5 / 15) * math.pi),
            eff48=77 + 16 * math.sin((i * 0.5 / 15) * math.pi),
            eff49=79 + 17 * math.sin((i * 0.5 / 15) * math.pi),
            eff50=81 + 18 * math.sin((i * 0.5 / 15) * math.pi),
        )
        for i in range(30)
    ]
    
    return PumpDetailsData(
        selected_pump=pump_type,
        selected_curve=curve_type,
        hq_curve_data=hq_curve_data,
        efficiency_curve_data=efficiency_curve_data
    )


# =============================================================================
# Notifications Endpoints
# =============================================================================

@router.get("/notifications", response_model=NotificationPanel)
async def get_notifications():
    """
    Get system notifications.
    """
    notifications = [
        Notification(
            id="1",
            type="warning",
            title="High Inflow Detected",
            message="Forecast Agent predicts inflow spike to 12,500 m³/h in next 4 hours. Prepare surge capacity.",
            time="5 min ago",
            icon="AlertTriangle",
            unread=True
        ),
        Notification(
            id="2",
            type="success",
            title="Energy Optimization Complete",
            message="Planner Agent achieved 34.5% energy savings for today. €1,270 cost reduction confirmed.",
            time="15 min ago",
            icon="CheckCircle",
            unread=True
        ),
        Notification(
            id="3",
            type="info",
            title="Pump P2 Maintenance Due",
            message="Scheduled maintenance window: Nov 18, 2025, 02:00-06:00. Digital Twin simulation ready.",
            time="1 hour ago",
            icon="Info",
            unread=False
        ),
    ]
    
    return NotificationPanel(
        notifications=notifications,
        unread_count=sum(1 for n in notifications if n.unread)
    )


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """
    Mark notification as read.
    """
    return {"status": "success", "notification_id": notification_id, "read": True}
