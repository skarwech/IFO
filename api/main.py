"""
FastAPI Backend for IFO System
-------------------------------
Main API application with REST endpoints and WebSocket support.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
import asyncio

from api.config import settings
from api.models import (
    SystemStatus,
    PumpStatus,
    TunnelMetrics,
    ForecastData,
    OptimizationRequest,
    OptimizationResult,
    HistoricalDataQuery,
    HistoricalDataResponse,
    KPIMetrics,
    ErrorResponse
)
from api.services import get_ifo_service
from api.websocket import get_connection_manager


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize services
ifo_service = get_ifo_service()
ws_manager = get_connection_manager()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting IFO Backend API...")
    ifo_service.initialize_agents()
    ws_manager.start_broadcasting(interval=2)
    print(f"✅ API ready at http://{settings.HOST}:{settings.PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down IFO Backend API...")
    ws_manager.stop_broadcasting()


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.API_VERSION
    }


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get current system status including all pumps and tunnel metrics."""
    try:
        return ifo_service.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pumps", response_model=List[PumpStatus])
async def get_pumps():
    """Get status of all pumps."""
    try:
        status = ifo_service.get_system_status()
        return status.pumps
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pumps/{pump_id}", response_model=PumpStatus)
async def get_pump(pump_id: int):
    """Get status of specific pump."""
    if pump_id < 1 or pump_id > 4:
        raise HTTPException(status_code=404, detail=f"Pump {pump_id} not found")
    
    try:
        status = ifo_service.get_system_status()
        return status.pumps[pump_id - 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tunnel", response_model=TunnelMetrics)
async def get_tunnel():
    """Get current tunnel metrics."""
    try:
        status = ifo_service.get_system_status()
        return status.tunnel
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast", response_model=ForecastData)
async def get_forecast(horizon: int = 96):
    """Get inflow forecast."""
    if horizon < 1 or horizon > settings.MAX_HORIZON:
        raise HTTPException(
            status_code=400,
            detail=f"Horizon must be between 1 and {settings.MAX_HORIZON}"
        )
    
    try:
        return ifo_service.get_forecast(horizon=horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize", response_model=OptimizationResult)
async def run_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Run optimization to generate optimal pump schedule.
    Can be run as background task for long horizons.
    """
    try:
        result = ifo_service.run_optimization(
            horizon=request.horizon,
            mode=request.mode
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/history", response_model=HistoricalDataResponse)
async def get_history(query: HistoricalDataQuery):
    """Get historical data for specified time range and metrics."""
    try:
        # Placeholder implementation
        timestamps = []
        current = query.start_time
        while current <= query.end_time:
            timestamps.append(current)
            current += timedelta(minutes=15)
        
        data = {
            metric: [0.0] * len(timestamps)
            for metric in query.metrics
        }
        
        return HistoricalDataResponse(
            timestamps=timestamps,
            data=data,
            metadata={
                "aggregation": query.aggregation or "raw",
                "count": len(timestamps)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kpis", response_model=KPIMetrics)
async def get_kpis(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
):
    """Get Key Performance Indicators for specified period."""
    if start is None:
        start = datetime.now() - timedelta(days=1)
    if end is None:
        end = datetime.now()
    
    try:
        return ifo_service.calculate_kpis(start=start, end=end)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pumps/{pump_id}/frequency")
async def set_pump_frequency(pump_id: int, frequency: float):
    """Manually set pump frequency (manual mode)."""
    if pump_id < 1 or pump_id > 4:
        raise HTTPException(status_code=404, detail=f"Pump {pump_id} not found")
    
    if frequency < 0 or frequency > 50:
        raise HTTPException(
            status_code=400,
            detail="Frequency must be between 0 and 50 Hz"
        )
    
    try:
        # Update pump frequency
        new_frequencies = ifo_service.current_frequencies.copy()
        new_frequencies[pump_id - 1] = frequency
        ifo_service.update_pump_frequencies(new_frequencies.tolist())
        
        return {"status": "success", "pump_id": pump_id, "frequency": frequency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time system updates.
    Broadcasts status every 2 seconds.
    """
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            # Echo back for heartbeat
            await websocket.send_json({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/api/docs",
        "health": "/api/health",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
