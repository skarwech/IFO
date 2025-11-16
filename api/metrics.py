"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps
from typing import Callable


# API request metrics
http_requests_total = Counter(
    'ifo_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'ifo_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Optimization metrics
optimization_runs_total = Counter(
    'ifo_optimization_runs_total',
    'Total optimization runs',
    ['status']
)

optimization_duration_seconds = Histogram(
    'ifo_optimization_duration_seconds',
    'Optimization duration in seconds',
    ['status']
)

optimization_energy_kwh = Histogram(
    'ifo_optimization_energy_kwh',
    'Optimized energy consumption in kWh',
    buckets=[1000, 2000, 5000, 10000, 20000, 50000]
)

# System metrics
tunnel_volume_m3 = Gauge(
    'ifo_tunnel_volume_m3',
    'Current tunnel volume in m³'
)

tunnel_level_m = Gauge(
    'ifo_tunnel_level_m',
    'Current tunnel level in meters'
)

pump_frequency_hz = Gauge(
    'ifo_pump_frequency_hz',
    'Current pump frequency in Hz',
    ['pump_id']
)

pump_power_kw = Gauge(
    'ifo_pump_power_kw',
    'Current pump power in kW',
    ['pump_id']
)

total_power_kw = Gauge(
    'ifo_total_power_kw',
    'Total system power in kW'
)

# WebSocket metrics
websocket_connections = Gauge(
    'ifo_websocket_connections',
    'Number of active WebSocket connections'
)

websocket_messages_sent = Counter(
    'ifo_websocket_messages_sent_total',
    'Total WebSocket messages sent'
)

# Chatbot metrics
chatbot_requests_total = Counter(
    'ifo_chatbot_requests_total',
    'Total chatbot requests',
    ['status']
)

chatbot_response_duration_seconds = Histogram(
    'ifo_chatbot_response_duration_seconds',
    'Chatbot response duration in seconds'
)


def track_http_metrics(method: str, endpoint: str):
    """Decorator to track HTTP request metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = getattr(e, 'status_code', 500)
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def update_system_metrics(status: dict):
    """Update system state metrics from status dict."""
    if 'tunnel' in status:
        tunnel = status['tunnel']
        tunnel_volume_m3.set(tunnel.get('volume', 0))
        tunnel_level_m.set(tunnel.get('level', 0))
    
    if 'pumps' in status:
        for pump in status['pumps']:
            pump_id = str(pump.get('id', 0))
            pump_frequency_hz.labels(pump_id=pump_id).set(pump.get('frequency', 0))
            pump_power_kw.labels(pump_id=pump_id).set(pump.get('power', 0))
    
    if 'total_power' in status:
        total_power_kw.set(status['total_power'])


def track_optimization_metrics(result: dict):
    """Track optimization result metrics."""
    status = result.get('status', 'unknown')
    optimization_runs_total.labels(status=status).inc()
    
    if 'computation_time' in result:
        optimization_duration_seconds.labels(status=status).observe(
            result['computation_time']
        )
    
    if 'total_energy' in result:
        optimization_energy_kwh.observe(result['total_energy'])
