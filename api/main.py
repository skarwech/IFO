"""
FastAPI Backend for IFO System
-------------------------------
Main API application with REST endpoints and WebSocket support.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
import json
import logging
import asyncio

from api.config import settings
from api.models import (
    SystemStatus,
    PumpStatus,
    TunnelMetrics,
    ForecastData,
    OptimizationRequest,
    OptimizationResult,
        FileUploadConfig,
        FileUploadResponse,
    HistoricalDataQuery,
    HistoricalDataResponse,
    KPIMetrics,
    ErrorResponse,
    EdgeMetricsPayload,
    ChatRequest,
    ChatResponse,
    ChatMessage,
)
from api.services import get_ifo_service
from api.websocket import get_connection_manager
from api.chatbot import get_aquabot
from api.file_handler import FileProcessor
from api.metrics import track_http_metrics
from api.security import verify_api_key, check_rate_limit
from api.metrics import (
    track_http_metrics,
    update_system_metrics,
    track_optimization_metrics,
)
from api.frontend_routes import router as frontend_router

try:
    from prometheus_client import make_asgi_app
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS for frontend (React app at http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Vite/React dev server
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Alternative Vite port
        "http://127.0.0.1:5173",
        *settings.CORS_ORIGINS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Service instances (initialized on startup)
ifo_service = None
ws_manager = None


def _get_service():
    """Get IFO service instance (must be initialized)."""
    if ifo_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return ifo_service


def _get_ws_manager():
    """Get WebSocket manager instance (must be initialized)."""
    if ws_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    return ws_manager


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global ifo_service, ws_manager
    
    print("🚀 Starting IFO Backend API...")
    
    # Initialize services
    ifo_service = get_ifo_service()
    ws_manager = get_connection_manager()
    
    # Link ws_manager to service
    ws_manager.set_service(ifo_service)
    
    # Initialize agents and WebSocket broadcasting
    ifo_service.initialize_agents()
    ws_manager.start_broadcasting(interval=2)
    
    # Include frontend routes
    app.include_router(frontend_router)
    print("🎨 Frontend API routes registered at /api/v1")
    
    # Add metrics endpoint if available
    if METRICS_ENABLED:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        print("📊 Prometheus metrics enabled at /metrics")
    
    print(f"✅ API ready at http://{settings.HOST}:{settings.PORT}")
    print(f"📖 API docs available at http://{settings.HOST}:{settings.PORT}/api/docs")
    print(f"🌐 Frontend integration ready for http://localhost:3000")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down IFO Backend API...")
    ws_manager.stop_broadcasting()


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/api/health")
@track_http_metrics("GET", "/api/health")
async def health_check():
    """Health check endpoint with detailed diagnostics."""
    service = _get_service()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.API_VERSION,
        "service_initialized": service is not None,
        "websocket_connections": ws_manager.active_connections if ws_manager else 0,
    }


@app.get("/api/status", response_model=SystemStatus)
@track_http_metrics("GET", "/api/status")
async def get_status():
    """Get current system status including all pumps and tunnel metrics."""
    try:
        service = _get_service()
        status = service.get_system_status()
        
        # Update metrics
        if METRICS_ENABLED:
            update_system_metrics(status.dict() if hasattr(status, 'dict') else status)
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pumps", response_model=List[PumpStatus])
async def get_pumps():
    """Get status of all pumps."""
    try:
        service = _get_service()
        status = service.get_system_status()
        return status.pumps
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pumps/{pump_id}", response_model=PumpStatus)
async def get_pump(pump_id: int):
    """Get status of specific pump."""
    if pump_id < 1 or pump_id > 4:
        raise HTTPException(status_code=404, detail=f"Pump {pump_id} not found")
    
    try:
        service = _get_service()
        status = service.get_system_status()
        return status.pumps[pump_id - 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tunnel", response_model=TunnelMetrics)
async def get_tunnel():
    """Get current tunnel metrics."""
    try:
        service = _get_service()
        status = service.get_system_status()
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
        service = _get_service()
        return service.get_forecast(horizon=horizon)
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
        service = _get_service()
        result = service.run_optimization(
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
        service = _get_service()
        return service.calculate_kpis(start=start, end=end)
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
        service = _get_service()
        new_frequencies = service.current_frequencies.copy()
        new_frequencies[pump_id - 1] = frequency
        service.update_pump_frequencies(new_frequencies.tolist())
        
        return {"status": "success", "pump_id": pump_id, "frequency": frequency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=FileUploadResponse)
@track_http_metrics("POST", "/api/upload")
async def upload_data_file(
    file: UploadFile = File(..., description="CSV or Excel file with inflow data"),
    config: str = Form(default='{}', description="JSON config for file processing")
):
    """
    Upload CSV or Excel file for forecasting and/or optimization.
    
    Allows you to:
    - Skip specific rows (0-based indexing)
    - Drop unwanted columns
    - Specify which column contains inflow data
    - Choose whether to forecast, optimize, or both
    
    Example config:
    {
        "skip_rows": [0, 1],
        "drop_columns": ["notes", "id"],
        "inflow_column": "flow_rate",
        "timestamp_column": "datetime",
        "use_for": "both"
    }
    """
    try:
        # Parse config
        try:
            config_dict = json.loads(config) if config else {}
            upload_config = FileUploadConfig(**config_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON config: {e}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {e}")
        
        # Process file
        df = await FileProcessor.read_uploaded_file(
            file,
            skip_rows=upload_config.skip_rows,
            drop_columns=upload_config.drop_columns
        )
        # Validate against company schema
        schema_check = FileProcessor.validate_dataset_columns(list(df.columns))
        if not schema_check.get("ok"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dataset does not match required company schema. "
                    f"Missing required: {schema_check.get('missing_required')}."
                )
            )
        
        # Extract inflow data
        inflow_values, timestamps, inflow_col_used = FileProcessor.extract_inflow_data(
            df,
            inflow_column=upload_config.inflow_column,
            timestamp_column=upload_config.timestamp_column
        )
        
        # Validate data
        FileProcessor.validate_inflow_data(inflow_values)
        
        # Get data preview
        preview = FileProcessor.get_data_preview(df)
        
        # Initialize results
        forecast_result = None
        optimization_result = None
        execution_result = None
        service = _get_service()
        
        # Update service with uploaded inflow data
        import numpy as np
        service.inflow_data = np.array(inflow_values)
        
        # Run full pipeline: Forecast → Plan → Execute
        if upload_config.use_for in ["forecast", "both"]:
            try:
                # Step 1: Forecast - Generate predictions
                forecast_result = service.get_forecast(horizon=96)
                log.info(f"✅ Forecast completed: {len(forecast_result.inflow_predictions)} predictions")
            except Exception as e:
                log.error(f"❌ Forecast failed: {e}")
        
        if upload_config.use_for in ["optimize", "both"]:
            try:
                # Step 2: Plan - Run MPC optimization
                optimization_result = service.run_optimization(horizon=min(96, len(inflow_values)))
                log.info(f"✅ Optimization completed: {optimization_result.status}")
                
                # Step 3: Execute - Apply first step of optimal schedule
                if optimization_result.status == "optimal" and optimization_result.schedule:
                    first_step_frequencies = optimization_result.schedule[0]
                    service.current_frequencies = np.array(first_step_frequencies)
                    
                    # Calculate execution metrics
                    total_flow = sum([
                        service.pump_fleet.small.flow_at_frequency(freq) * 3.6 
                        for freq in first_step_frequencies
                    ])
                    total_power = sum([
                        service.pump_fleet.small.power_at_frequency(freq)
                        for freq in first_step_frequencies
                    ])
                    
                    execution_result = {
                        "applied_frequencies": first_step_frequencies,
                        "total_flow_m3h": round(total_flow, 2),
                        "total_power_kw": round(total_power, 2),
                        "execution_time": datetime.now().isoformat()
                    }
                    log.info(f"✅ Execution completed: Applied frequencies {first_step_frequencies}")
                
            except Exception as e:
                log.error(f"❌ Optimization/Execution failed: {e}")
        
        # Build response message
        pipeline_steps = []
        if forecast_result:
            pipeline_steps.append("Forecast ✅")
        if optimization_result:
            pipeline_steps.append("Plan ✅")
        if execution_result:
            pipeline_steps.append("Execute ✅")
        
        pipeline_summary = " → ".join(pipeline_steps) if pipeline_steps else "No pipeline steps run"
        
        return FileUploadResponse(
            status="success",
            rows_processed=len(df),
            columns_found=list(df.columns),
            data_preview=preview,
            forecast_result=forecast_result,
            optimization_result=optimization_result,
            execution_result=execution_result,
            message=(
                f"Processed {len(inflow_values)} inflow data points from '{file.filename}'. "
                f"Inflow column used: '{inflow_col_used}'. "
                f"Schema validated: OK; Missing optional: {schema_check.get('missing_optional')}. "
                f"Pipeline: {pipeline_summary}"
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"File upload error: {e}")
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


# ============================================================================
# Edge Integration Endpoints
# ============================================================================

@app.post("/api/edge/metrics")
async def post_edge_metrics(payload: EdgeMetricsPayload):
    """Receive metrics from ARM/edge device."""
    try:
        service = _get_service()
        service.update_edge_metrics(payload.metrics)
        return {"status": "ok", "received_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/edge/metrics")
async def get_edge_metrics():
    """Return last metrics reported by edge device."""
    try:
        service = _get_service()
        return service.get_edge_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Chatbot Endpoints
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_aquabot(request: ChatRequest) -> ChatResponse:
    """
    Chat with AquaBot, the Gemini-powered assistant for IFO system.
    
    Provides context-aware responses about:
    - System status and optimization
    - Pump operations and best practices
    - Troubleshooting and configuration
    - Energy savings and KPIs
    """
    try:
        bot = get_aquabot()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"AquaBot unavailable. Check GEMINI_API_KEY is set. Error: {str(e)}"
        )
    
    # Get system status if requested
    system_status = None
    if request.include_system_status:
        try:
            service = _get_service()
            status = service.get_system_status()
            system_status = {
                "total_power": status.total_power,
                "optimization_active": status.optimization_active,
                "tunnel": {
                    "volume": status.tunnel.volume,
                    "level": status.tunnel.level,
                },
                "pumps": [
                    {"id": p.id, "frequency": p.frequency, "is_running": p.is_running}
                    for p in status.pumps
                ],
            }
        except Exception as e:
            print(f"Warning: Failed to get system status for chat context: {e}")
    
    # Generate response
    response_text = bot.chat(request.message, system_status=system_status)
    
    return ChatResponse(response=response_text)


@app.delete("/api/chat/history", tags=["Chatbot"])
async def reset_chat_history():
    """
    Reset AquaBot conversation history.
    """
    try:
        bot = get_aquabot()
        bot.reset_conversation()
        return {"status": "success", "message": "Conversation history reset"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/chat/history", response_model=list[ChatMessage], tags=["Chatbot"])
async def get_chat_history() -> list[ChatMessage]:
    """
    Get AquaBot conversation history.
    """
    try:
        bot = get_aquabot()
        history = bot.get_conversation_history()
        messages = []
        for entry in history:
            messages.append(ChatMessage(
                role="user",
                content=entry["user"],
                timestamp=datetime.fromisoformat(entry["timestamp"])
            ))
            messages.append(ChatMessage(
                role="assistant",
                content=entry["assistant"],
                timestamp=datetime.fromisoformat(entry["timestamp"])
            ))
        return messages
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )

log = logging.getLogger("api")
