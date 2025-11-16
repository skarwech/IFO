# IFO: Intelligent Flood Optimization
**Multi-Agent AI + MPC + Digital Twin for Wastewater Pump Optimization**

Junction 2025 Competition • Blominmäki WWTP (HSY Finland)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [API Backend](#api-backend)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Frontend Integration](#frontend-integration)
- [Mathematical Formulation](#mathematical-formulation)
- [File Structure](#file-structure)
- [Development](#development)
- [Testing](#testing)
- [Security](#security)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

IFO is a sophisticated multi-agent system leveraging Model Predictive Control (MPC) to optimize pump operations in urban stormwater management, minimizing energy costs while maintaining safe tunnel volumes.

### Key Features

- **Multi-Agent Architecture**: ForecastAgent (LSTM) → PlannerAgent (MILP/MPC) → ExecutorAgent (physics) → SupervisorAgent (KPIs)
- **Enhanced Pump Models**: Digitized Grundfos curves with affinity laws (Q∝f, H∝f², P∝f³)
- **MPC Optimization**: Discrete frequencies {48, 49, 50} Hz with terminal balance and smoothing
- **LSTM Forecasting**: 32-step lookback, 96-step horizon with persistence fallback
- **REST + WebSocket API**: Production-ready FastAPI backend for real-time integration
- **Digital Twin**: OPC UA server/client for industrial integration
- **Docker Ready**: Containerized deployment with docker-compose

### Performance

- **Cost Savings**: 10-30% typical energy cost reduction
- **Reliability**: 0 constraint violations, stable operation
- **Optimization**: 96-step horizon in <10 seconds
- **Real-time**: 2-second WebSocket updates

---

## Quick Start

### 1. Local API Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (edit .env with your settings)
# Optional: add GEMINI_API_KEY for chatbot

# Start the API server
uvicorn api.main:app --reload

# Access API documentation
# Browser: http://localhost:8000/api/docs
```

### 2. Frontend Integration (Aquaoptai)

```bash
# Start the backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test frontend API endpoints
python test_frontend_api.py

# Configure your React frontend to use:
# API_BASE_URL: http://localhost:8000
# WS_URL: ws://localhost:8000/ws/live
```

### 3. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# API available at http://localhost:8000
```

### 3. Run Tests

```bash
# Install test tools (already included in requirements.txt)
pip install -r requirements.txt

# Run test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=api --cov-report=html
```

### 4. Use AquaBot (Optional)

```bash
# Set your Gemini API key (get one at https://ai.google.dev/)
export GEMINI_API_KEY=your_api_key_here  # Linux/Mac
# or
$env:GEMINI_API_KEY="your_api_key_here"  # Windows PowerShell

# Start API and chat at http://localhost:8000/api/docs
# Try POST /api/chat with: {"message": "What is IFO?"}
```

---

## API Backend

### Endpoints

#### System Status
- **GET** `/api/health` - Health check
- **GET** `/api/status` - Complete system status (pumps + tunnel)
- **GET** `/api/pumps` - All pump statuses
- **GET** `/api/pumps/{id}` - Individual pump status
- **GET** `/api/tunnel` - Current tunnel metrics

#### Forecasting & Optimization
- **GET** `/api/forecast?horizon=96` - Inflow predictions
- **POST** `/api/optimize` - Run MPC optimization

#### Analytics
- **POST** `/api/history` - Historical data query
- **GET** `/api/kpis` - Key Performance Indicators

#### Control
- **POST** `/api/pumps/{id}/frequency` - Manual pump control

#### Chatbot (AquaBot)
- **POST** `/api/chat` - Chat with Gemini-powered assistant
- **GET** `/api/chat/history` - Get conversation history
- **DELETE** `/api/chat/history` - Reset conversation

#### Edge/ARM
- **POST** `/api/edge/metrics` - Receive edge device metrics
- **GET** `/api/edge/metrics` - Get latest edge metrics

#### Real-Time
- **WS** `/ws/live` - WebSocket streaming (2s updates)

### Example Responses

#### System Status
```json
{
  "pumps": [
    {"id": 1, "frequency": 48.0, "flow": 120.5, "power": 95.3, "mode": "auto", "is_running": true}
  ],
  "tunnel": {
    "volume": 5000.0,
    "level": 2.5,
    "inflow_rate": 100.0,
    "outflow_rate": 450.0
  },
  "total_power": 380.2,
  "optimization_active": false
}
```

#### Optimization Result
```json
{
  "schedule": [[48, 48, 48, 48], ...],
  "predicted_power": [400.2, 385.1, ...],
  "predicted_volume": [5000, 4950, ...],
  "total_energy": 2400.5,
  "cost_savings": 15.5,
  "computation_time": 2.3,
  "status": "optimal"
}
```

#### AquaBot Chat Response
```json
{
  "response": "MPC (Model Predictive Control) optimizes pump schedules by solving a MILP problem over a 96-step horizon (24 hours). It uses LSTM forecasts for inflow, applies affinity laws (P∝f³) for energy, and enforces volume constraints while minimizing electricity cost. The optimizer runs in <10s and provides cost savings of 10-30%.",
  "timestamp": "2025-11-15T14:30:00Z"
}
```

---

## System Architecture

### Multi-Agent Design

```
┌─────────────────────────────────────────────────┐
│          AquaOptAI Frontend (Next.js)           │
│  ┌─────────────┐         ┌──────────────────┐  │
│  │  Dashboard  │         │  Optimization UI │  │
│  └─────────────┘         └──────────────────┘  │
└────────────┬──────────────────────┬─────────────┘
             │ REST API             │ WebSocket
             │                      │
┌────────────┴──────────────────────┴─────────────┐
│           IFO FastAPI Backend                    │
│  ┌──────────┐  ┌─────────┐  ┌────────────────┐ │
│  │ main.py  │  │ models  │  │  websocket.py  │ │
│  └────┬─────┘  └─────────┘  └────────────────┘ │
│       │                                          │
│  ┌────┴──────────────────────────────────────┐  │
│  │         services.py (IFOService)          │  │
│  └────┬──────────────────────────────────────┘  │
└───────┼──────────────────────────────────────────┘
        │
┌───────┴──────────────────────────────────────────┐
│         Multi-Agent System (src/agents.py)       │
│                                                   │
│  ┌──────────────┐      ┌─────────────────────┐  │
│  │ForecastAgent │──────│  PlannerAgent (MPC) │  │
│  │    (LSTM)    │      │       (MILP)        │  │
│  └──────────────┘      └──────────┬──────────┘  │
│                                   │              │
│  ┌──────────────┐      ┌──────────┴──────────┐  │
│  │ExecutorAgent │──────│  SupervisorAgent    │  │
│  │  (Physics)   │      │   (KPIs/Alerts)     │  │
│  └──────────────┘      └─────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Agent Responsibilities

1. **ForecastAgent**: LSTM-based inflow prediction with 96-step horizon
2. **PlannerAgent**: MILP/MPC optimization with discrete frequencies
3. **ExecutorAgent**: Physics-based simulation and state propagation
4. **SupervisorAgent**: KPI monitoring, alerts, and drift detection

---

## Development

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- pip package manager
- CBC solver (for MILP optimization)
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Install dependencies (includes dev/test tools)
pip install -r requirements.txt

# Start development server with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Access API docs
# Browser: http://localhost:8000/api/docs
```

### Core Dependencies

- **FastAPI**: Web framework for API
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **PyTorch**: LSTM forecasting
- **PuLP**: MILP optimization
- **NumPy/Pandas**: Data processing
- **Plotly/Dash**: Visualization
- **Gemini API**: Chatbot (google-generativeai)

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_optimization.py -v

# With coverage report
pytest tests/ --cov=src --cov=api --cov-report=html

# Test frontend API endpoints
python test_frontend_api.py
```

### Test Structure

- **tests/test_models.py**: Pump models and system dynamics
- **tests/test_optimization.py**: MPC optimizer and constraints
- **tests/test_api.py**: API endpoints and WebSocket
- **tests/conftest.py**: Shared test fixtures

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Use fixtures from `conftest.py` for common setup
- Mock external dependencies (OPC UA, APIs)

Example:
```python
def test_optimizer_respects_volume_constraints(sample_config, sample_inflow_data):
    """Test that MPC optimizer keeps volume within bounds."""
    optimizer = MPCOptimizer(sample_config)
    result = optimizer.optimize(sample_inflow_data)
    
    assert result['status'] == 'optimal'
    assert all(V_min <= v <= V_max for v in result['volumes'])
```

---

## Security

### Reporting Vulnerabilities

If you discover a security vulnerability:
1. **DO NOT** create a public GitHub issue
2. Report via GitHub Security Advisories
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge within 48 hours and respond within 7 days.

### Production Security Best Practices

#### 1. Enable Authentication
```bash
ENABLE_AUTH=true
API_KEY=<strong-random-key>
```

Generate strong API keys:
```python
import secrets
api_key = secrets.token_urlsafe(32)
print(f"API_KEY={api_key}")
```

#### 2. Use HTTPS
- Deploy behind HTTPS reverse proxy
- Use Let's Encrypt for certificates

#### 3. Environment Variables
- Never commit `.env` to version control
- Use secrets management (AWS Secrets Manager, Azure Key Vault)
- Rotate keys regularly

#### 4. Rate Limiting
```bash
ENABLE_RATE_LIMIT=true
```

#### 5. Network Security
- Restrict OPC UA access to internal network
- Use firewall rules
- VPN for remote access

#### 6. Regular Updates
- Keep dependencies updated
- Monitor security advisories
- Scan Docker images for vulnerabilities

### Known Security Considerations

**OPC UA Communication:**
- Currently uses anonymous authentication
- For production: Enable certificate-based authentication
- Encrypt OPC UA traffic

**WebSocket Connections:**
- Unauthenticated by default
- Add authentication for production

**Chatbot (AquaBot):**
- Gemini API key stored in environment
- Rate limiting applied
- No sensitive data sent to Gemini

### Docker Security

- Run containers as non-root user
- Use minimal base images
- Scan images for vulnerabilities
- Keep images updated

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Active support  |
| < 1.0   | ❌ Not supported   |

---

## Deployment

### 1. Start API Server

```bash
# Development (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 2. CLI Mode (Legacy)

```bash
# Single optimization run
python main.py data/test_data.csv data/test_volume.csv --horizon 24

# With dashboard
python main.py data/test_data.csv data/test_volume.csv --dashboard
```

### 3. Multi-Agent Mode

```bash
# Offline mode (historical data)
python main_multiagent.py --mode offline --data data/test_data.csv --steps 96 --report

# Realtime mode (with OPC UA)
python main_multiagent.py --mode realtime --steps 96

# Hybrid mode (RL-gated MPC)
python main.py data/test_data.csv data/test_volume.csv --hybrid
```

### 4. Validation

```bash
# Run baseline calculation
python calculate_baseline.py

# Full validation suite
python validate_junction2025.py
```

### 5. AquaBot Chatbot

```bash
# Set API key in environment
export GEMINI_API_KEY=your_key  # or use .env file

# Use via API (see /api/docs)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does MPC optimization work?"}'

# Response includes system status context automatically
# Reset conversation: DELETE /api/chat/history
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# API Settings
API_TITLE=IFO Backend API
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=INFO

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:3000,https://aquaoptai.vercel.app

# Gemini API for AquaBot
GEMINI_API_KEY=your_google_api_key_here

# IFO Configuration
IFO_CONFIG_PATH=./config.yaml
DEFAULT_HORIZON=96
MAX_HORIZON=288

# WebSocket
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100
```

### System Configuration (config.yaml)

```yaml
timestep_minutes: 15
horizon_steps: 96

tunnel:
  initial_volume: 5000.0
  max_capacity: 10000.0
  min_level: 0.0
  max_level: 8.0

optimization:
  horizon: 96
  objective: minimize_cost
  solver_timeout: 10
  solver_gap: 0.02

forecasting:
  model: lstm
  lookback: 24
  epochs: 50

opcua:
  endpoint: opc.tcp://localhost:4840
```

---

## API Documentation

### Accessing Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

### WebSocket Usage

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onopen = () => console.log('Connected');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
  // {timestamp, pumps, tunnel, total_power, event_type}
};

// Send heartbeat
setInterval(() => ws.send('ping'), 30000);
```

---

## Frontend Integration

The IFO backend provides comprehensive API endpoints designed specifically for the [Aquaoptai](https://github.com/skarwech/Aquaoptai) React frontend.

### Architecture

The backend provides two types of APIs:

1. **Legacy API** (`/api/*`): Original endpoints for optimization, forecasting, and system control
2. **Frontend API** (`/api/v1/*`): New endpoints specifically designed for the Aquaoptai React components

### Quick Start

```bash
# 1. Start the backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 2. Test the frontend APIs
python test_frontend_api.py

# 3. Configure your React app
# .env.local or src/config.ts:
# API_BASE_URL=http://localhost:8000
# WS_URL=ws://localhost:8000/ws/live
```

### CORS Configuration

The backend is pre-configured to accept requests from:
- `http://localhost:3000` (Create React App)
- `http://localhost:5173` (Vite)
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

To add custom origins, update your `.env` file:
```bash
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://your-custom-port
```

### Frontend API Endpoints (v1)

All frontend endpoints are under `/api/v1/`:

#### Dashboard
- **GET** `/api/v1/dashboard` - Complete dashboard data (cards, pumps, alerts, flow/price data)

#### Agent Views
- **GET** `/api/v1/agents/forecast?horizon=24` - Forecast agent (inflow predictions, prices)
- **GET** `/api/v1/agents/planner?price_scenario=normal` - Planner agent (pump schedules, costs)
- **GET** `/api/v1/agents/executor?is_executing=true` - Executor agent (real-time control)
- **GET** `/api/v1/agents/supervisor` - Supervisor agent (metrics, communications, constraints)

#### Analytics & Reports
- **GET** `/api/v1/simulations?scenario=normal` - Simulation comparisons
- **GET** `/api/v1/reports?time_range=week` - Savings analysis and KPIs
- **GET** `/api/v1/system/overview` - System architecture overview

#### Configuration
- **GET** `/api/v1/settings` - System configuration
- **POST** `/api/v1/settings/opcua/test` - Test OPC UA connection
- **GET** `/api/v1/pumps/details?pump_type=small` - Pump performance curves

#### Notifications
- **GET** `/api/v1/notifications` - System notifications
- **POST** `/api/v1/notifications/{id}/read` - Mark notification as read

### WebSocket Real-Time Updates

Connect to WebSocket for live system updates:

```typescript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'system_update') {
    // Update dashboard with latest data
    updateDashboard(data.dashboard);
    updatePumps(data.pumps);
    updateTunnel(data.tunnel);
  }
};

// Send heartbeat
setInterval(() => {
  ws.send(JSON.stringify({ type: 'ping' }));
}, 30000);
```

**Message Format:**
```json
{
  "type": "system_update",
  "timestamp": "2024-11-15T10:30:00",
  "dashboard": {
    "currentPower": 847.0,
    "tunnelLevel": 4.2,
    "tunnelVolume": 156800,
    "inflowRate": 10234,
    "outflowRate": 9876,
    "activePumps": 4
  },
  "pumps": [...],
  "tunnel": {...},
  "total_power": 847.0,
  "event_type": "status_update"
}
```

### React Integration Examples

#### Using Axios

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 10000,
});

// Dashboard component
const Dashboard = () => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        const response = await api.get('/dashboard');
        setData(response.data);
      } catch (error) {
        console.error('Failed to fetch dashboard:', error);
      }
    };
    
    fetchDashboard();
    const interval = setInterval(fetchDashboard, 5000); // Poll every 5s
    
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div>
      {data?.cards.map(card => (
        <MetricCard key={card.title} {...card} />
      ))}
    </div>
  );
};
```

#### Using WebSocket Hook

```typescript
const useWebSocket = (url: string) => {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setData(message);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };
    
    return () => ws.close();
  }, [url]);
  
  return { data, connected };
};

// Usage
const LiveDashboard = () => {
  const { data, connected } = useWebSocket('ws://localhost:8000/ws/live');
  
  return (
    <div>
      <StatusIndicator connected={connected} />
      {data?.dashboard && <Metrics data={data.dashboard} />}
    </div>
  );
};
```

### Troubleshooting Frontend Integration

**CORS Errors:**
- Verify the frontend URL is in `CORS_ORIGINS` environment variable
- Restart the backend after changing CORS settings
- Check browser dev tools for the exact origin being blocked

**WebSocket Connection Failed:**
- Ensure backend is running on the correct port
- Check firewall settings
- Use `ws://` not `wss://` for local development

**404 Not Found:**
- Verify you're using the correct API prefix: `/api/v1/`
- Check the backend logs for registered routes
- Visit `/api/docs` to see all available endpoints

**Data Not Updating:**
- Check WebSocket connection status
- Verify polling interval in frontend code
- Check backend logs for errors
- Test endpoints directly with `curl` or Postman

---

### React/Next.js Example

#### 1. API Client

```typescript
// lib/ifo-client.ts
const API_BASE = process.env.NEXT_PUBLIC_IFO_API || 'http://localhost:8000';

export const ifoClient = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/api/status`);
    return res.json();
  },
  
  async getForecast(horizon: number = 96) {
    const res = await fetch(`${API_BASE}/api/forecast?horizon=${horizon}`);
    return res.json();
  },
  
  async optimize(config: { horizon: number; mode: string }) {
    const res = await fetch(`${API_BASE}/api/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return res.json();
  },
};
```

#### 2. WebSocket Hook

```typescript
// hooks/useIFOWebSocket.ts
import { useEffect, useState } from 'react';

export function useIFOWebSocket() {
  const [status, setStatus] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/live');

    ws.onopen = () => setConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.event_type === 'status_update') setStatus(data);
    };
    ws.onclose = () => setConnected(false);

    return () => ws.close();
  }, []);

  return { status, connected };
}
```

#### 3. Component Usage

```typescript
'use client';

import { useIFOWebSocket } from '@/hooks/useIFOWebSocket';

export default function Dashboard() {
  const { status, connected } = useIFOWebSocket();

  return (
    <div>
      <div className="status">
        {connected ? '🟢 Live' : '🔴 Disconnected'}
      </div>
      {status && (
        <div>
          <h2>Total Power: {status.total_power.toFixed(1)} kW</h2>
          {status.pumps.map(pump => (
            <div key={pump.id}>
              Pump {pump.id}: {pump.frequency} Hz, {pump.power.toFixed(1)} kW
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

#### 4. AquaBot Chatbot

```typescript
'use client';

import { useState } from 'react';

export default function ChatBot() {
  const [message, setMessage] = useState('');
  const [chat, setChat] = useState<Array<{role: string; content: string}>>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;
    
    setChat(prev => [...prev, { role: 'user', content: message }]);
    setLoading(true);
    
    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, include_system_status: true }),
      });
      const data = await res.json();
      setChat(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err) {
      console.error('Chat error:', err);
    } finally {
      setLoading(false);
      setMessage('');
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {chat.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'AquaBot'}:</strong>
            <p>{msg.content}</p>
          </div>
        ))}
      </div>
      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
        placeholder="Ask AquaBot about IFO..."
        disabled={loading}
      />
      <button onClick={sendMessage} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>
    </div>
  );
}
```

---

## Mathematical Formulation

### System Dynamics

```
V(t+1) = V(t) + F1(t)·Δt - F2(t)·Δt
L(t) = f(V(t))  [volume-to-level mapping]
Δt = 0.25 hours (15 minutes)
```

### Pump Models (Affinity Laws)

```
Q(f) ∝ f        [Flow proportional to frequency]
H(f) ∝ f²       [Head proportional to frequency squared]
P(f) ∝ f³       [Power proportional to frequency cubed]
```

### MPC Objective

```
minimize: Σ[t=0 to T-1] P(t)·price(t)·Δt + penalties

penalties:
  - Terminal balance: |V(T) - V(0)|
  - Average flow constraint: avg(F2) ≥ avg(F1)
  - Smoothing: Σ|F2(t) - F2(t-1)|
  - Frequency constancy: Σ|freq(t) - freq(t-1)|
```

### Constraints

```
V_min ≤ V(t) ≤ V_max           [Volume bounds]
L_min ≤ L(t) ≤ L_max           [Level bounds]
|F2(t) - F2(t-1)| ≤ ramp_limit [Ramp limit]
Σ y_pump(t) ≥ 1                [At least 1 pump on]
freq ∈ {48, 49, 50} Hz         [Discrete frequencies]
```

---

## File Structure

```
IFO/
├── api/                      # FastAPI Backend
│   ├── __init__.py
│   ├── main.py              # API routes and app
│   ├── models.py            # Pydantic schemas
│   ├── services.py          # Business logic
│   ├── websocket.py         # WebSocket manager
│   ├── chatbot.py           # AquaBot (Gemini)
│   └── config.py            # Settings
├── src/                      # Core IFO System
│   ├── agents.py            # Multi-agent framework
│   ├── optimize.py          # MILP/MPC optimizer
│   ├── forecast.py          # LSTM forecasting
│   ├── model.py             # Pump curves & models
│   ├── digital_twin.py      # OPC UA integration
│   ├── dashboard_multiagent.py  # Visualization
│   └── data_utils.py        # Data processing
├── data/                     # Data files
│   ├── test_data.csv
│   └── test_volume.csv
├── results/                  # Output files
│   └── README.md
├── arm/                      # Edge/ARM module
│   ├── __init__.py
│   ├── edge_agent.py
│   ├── opcua_edge.py
│   ├── metrics.py
│   └── deploy.ps1
├── main.py                   # CLI entry point (legacy)
├── main_multiagent.py       # Multi-agent CLI
├── calculate_baseline.py    # Baseline metrics
├── validate_junction2025.py # Validation suite
├── test_api.py              # API test suite
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image
├── docker-compose.yml       # Multi-service setup
├── .env                     # Environment configuration (gitignored)
└── README.md                # This file
```

---

## Development

### Local Development Environment

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/IFO.git
cd IFO

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (includes dev/test tools)
pip install -r requirements.txt

# Configure environment
# Edit .env with your settings

# Start development server with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Code Style Guidelines

- Follow PEP 8 (max line length: 127)
- Use Black for formatting: `black src api tests`
- Use flake8 for linting: `flake8 src api tests`
- Type hints required for public functions
- Add docstrings for public functions

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_optimization.py -v

# With coverage report
pytest tests/ --cov=src --cov=api --cov-report=html

# Test frontend API endpoints
python test_frontend_api.py
```

### Test Structure

- **tests/test_models.py**: Pump models and system dynamics
- **tests/test_optimization.py**: MPC optimizer and constraints
- **tests/test_api.py**: API endpoints and WebSocket
- **tests/conftest.py**: Shared test fixtures

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Use fixtures from `conftest.py` for common setup
- Mock external dependencies (OPC UA, APIs)

Example:
```python
def test_optimizer_respects_volume_constraints(sample_config, sample_inflow_data):
    """Test that MPC optimizer keeps volume within bounds."""
    optimizer = MPCOptimizer(sample_config)
    result = optimizer.optimize(sample_inflow_data)
    
    assert result['status'] == 'optimal'
    assert all(V_min <= v <= V_max for v in result['volumes'])
```

---

## Security

### Reporting Vulnerabilities

If you discover a security vulnerability:
1. **DO NOT** create a public GitHub issue
2. Report via GitHub Security Advisories
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge within 48 hours and respond within 7 days.

### Production Security Best Practices

#### 1. Enable Authentication
```bash
ENABLE_AUTH=true
API_KEY=<strong-random-key>
```

Generate strong API keys:
```python
import secrets
api_key = secrets.token_urlsafe(32)
print(f"API_KEY={api_key}")
```

#### 2. Use HTTPS
- Deploy behind HTTPS reverse proxy
- Use Let's Encrypt for certificates

#### 3. Environment Variables
- Never commit `.env` to version control
- Use secrets management (AWS Secrets Manager, Azure Key Vault)
- Rotate keys regularly

#### 4. Rate Limiting
```bash
ENABLE_RATE_LIMIT=true
```

#### 5. Network Security
- Restrict OPC UA access to internal network
- Use firewall rules
- VPN for remote access

#### 6. Regular Updates
- Keep dependencies updated
- Monitor security advisories
- Scan Docker images for vulnerabilities

### Known Security Considerations

**OPC UA Communication:**
- Currently uses anonymous authentication
- For production: Enable certificate-based authentication
- Encrypt OPC UA traffic

**WebSocket Connections:**
- Unauthenticated by default
- Add authentication for production

**Chatbot (AquaBot):**
- Gemini API key stored in environment
- Rate limiting applied
- No sensitive data sent to Gemini

### Docker Security

- Run containers as non-root user
- Use minimal base images
- Scan images for vulnerabilities
- Keep images updated

### Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Active support  |
| < 1.0   | ❌ Not supported   |

---

## Deployment

### ARM/Edge Deployment

Deploy the edge control loop to an ARM device (e.g., Raspberry Pi) and connect it to the backend.

#### Prerequisites

- ARM device with Linux (Debian/Raspberry Pi OS/Ubuntu) and Python 3.8+
- Network access to the backend API (default `http://<server>:8000`)
- Optional: TorchScript policy file (`.pt`) for on-device inference

#### Sync from Windows (PowerShell)

Use the provided script to sync code and set up a virtual environment on the device.

```powershell
# From repo root on your Windows machine
# Example: deploy to raspberrypi.local with user 'pi' into /home/pi/ifo
powershell -ExecutionPolicy Bypass -File .\arm\deploy.ps1 -Host pi@raspberrypi.local -Dest "/home/pi/ifo" -Python "python3" -Venv ".venv"
```

Parameters:
- `-Host`: SSH host (e.g., `pi@raspberrypi.local` or `user@192.168.1.50`)
- `-Dest`: Destination folder on the device
- `-Python`: Python executable on device (default `python3`)
- `-Venv`: Virtual env folder name (default `.venv`)

If you prefer manual setup, copy the `arm/` and `requirements.txt` and then create a venv and install deps on the device.

#### Run Edge Agent (on the device)

```bash
cd /home/pi/ifo
source .venv/bin/activate  # or 'source venv/bin/activate'
python -m arm.edge_agent \
  --backend http://<server>:8000 \
  --opcua opc.tcp://<opc-server>:4840 \
  --period 5 \
  --hybrid --horizon 24      # optional: hybrid MPC from backend
# Optional: add a TorchScript policy
#   --policy /home/pi/models/ifo_policy.pt
```

Behavior:
- Reads state from OPC UA server, writes pump frequencies back
- Posts lightweight device metrics to backend: `POST /api/edge/metrics`
- Optional hybrid loop: calls backend `POST /api/optimize` and applies first-step schedule

#### Configure OPC UA NodeIds

Edit `arm/opcua_edge.py` to match your server model.
- Default uses namespace 2 string NodeIds like `ns=2;s=Pump1_Hz`, `ns=2;s=Level_m`
- Update in `read_state()` and `write_pumps()` as needed

#### Backend Endpoints for Edge

- `POST /api/edge/metrics`: body `{ "metrics": {"cpu_temp_c": 55.1, ... } }`
- `GET /api/edge/metrics`: returns latest payload plus server timestamp

#### Edge Troubleshooting

- Install missing deps on device: `pip install -r requirements.txt`
- Verify OPC UA connectivity with a test client; adjust NodeIds
- Check backend reachability: `curl http://<server>:8000/api/health`
- Reduce `--period` if the loop is too slow; increase if too chatty

### Docker Deployment

```bash
# Build image
docker build -t ifo-api .

# Run container
docker run -p 8000:8000 --env-file .env ifo-api

# Or use Docker Compose
docker-compose up -d
```

### Cloud Deployment

#### AWS (ECS/Fargate)

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag ifo-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ifo-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ifo-api:latest

# Deploy to ECS (via AWS Console or CLI)
```

#### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/<project-id>/ifo-api
gcloud run deploy ifo-api --image gcr.io/<project-id>/ifo-api --platform managed --region us-central1 --allow-unauthenticated
```

#### Azure Container Instances

```bash
az container create --resource-group ifo-rg --name ifo-api --image <registry>/ifo-api:latest --cpu 1 --memory 2 --ports 8000
```

### Production Checklist

- [ ] Set `RELOAD=false` in production
- [ ] Configure CORS for production domains
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and logging
- [ ] Configure auto-scaling
- [ ] Set up health checks
- [ ] Enable rate limiting
- [ ] Secure API keys/secrets

---

## Troubleshooting

### API Issues

**Import errors:**
```bash
pip install -r requirements.txt
python -c "from api.main import app; print('OK')"
```

**Port already in use:**
```bash
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**WebSocket connection fails:**
- Check CORS settings include WebSocket protocol
- Verify firewall allows WebSocket connections
- Use `ws://` for local, `wss://` for HTTPS

### Optimization Issues

**Infeasible solutions:**
- Check initial volume within bounds
- Reduce horizon or loosen constraints
- Verify inflow forecast data

**Slow optimization:**
- Decrease `horizon_steps` in config
- Increase `solver_timeout` (default: 10s)
- Accept feasible solutions (not just optimal)

### LSTM Issues

**Training fails:**
- Reduce `epochs` or `lookback_steps`
- Ensure ≥32 historical samples
- Use persistence fallback if needed

**Poor predictions:**
- Increase training data
- Tune LSTM hyperparameters
- Check data quality/stationarity

---

## License

MIT License

Copyright (c) 2025 IFO Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. **Clone and Setup**
   ```bash
   git clone https://github.com/YOUR_USERNAME/IFO.git
   cd IFO
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Edit .env with your settings
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   pytest tests/ --cov=src --cov=api --cov-report=html
   ```

### Code Style

- Follow PEP 8 (max line length: 127)
- Use Black for formatting: `black src api tests`
- Use flake8 for linting: `flake8 src api tests`
- Type hints required for public functions
- Add docstrings for public functions

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code with tests
   - Update documentation
   - Follow commit conventions:
     - `feat:` New feature
     - `fix:` Bug fix
     - `docs:` Documentation only
     - `test:` Adding tests
     - `refactor:` Code refactoring
     - `perf:` Performance improvement
     - `chore:` Maintenance tasks

3. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR Review**
   - Ensure CI passes (tests, linting, type checks)
   - Address review feedback
   - Keep commits clean and atomic

### Common Tasks

**Adding API Endpoint:**
1. Define route in `api/main.py` or `api/frontend_routes.py`
2. Add Pydantic models in `api/models.py` or `api/frontend_models.py`
3. Implement logic in `api/services.py` or `src/`
4. Add tests in `tests/test_api.py`

**Adding Optimization Constraint:**
1. Update `src/optimize.py`
2. Add constraint to `MPCOptimizer._add_constraints()`
3. Test in `tests/test_optimization.py`

**Adding Pump Model:**
1. Define model in `src/pump_models.py`
2. Add tests in `tests/test_models.py`
3. Update configuration schema in `config.yaml`

### Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

---

## License

**IFO Team** • Junction 2025 Valmet–HSY Challenge

- Repository: https://github.com/skarwech/IFO
- Frontend: https://github.com/skarwech/AquaOptAI
- Issues: https://github.com/skarwech/IFO/issues

For support, open an issue with:
- Environment details (OS, Python version)
- Error messages and logs
- Sample data snippet (if applicable)

---

**Built with ❤️ for sustainable water management**

