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
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
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

# Start the API server
uvicorn api.main:app --reload

# Access API documentation
# Browser: http://localhost:8000/api/docs
```

### 2. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# API available at http://localhost:8000
```

### 3. Test the API

```bash
# Run test suite
python test_api.py

# Should see: ✅ 7/7 tests passing
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

## Installation

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- pip package manager
- CBC solver (for MILP optimization)
- Docker (optional, for containerized deployment)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Core Dependencies

- **FastAPI**: Web framework for API
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **PyTorch**: LSTM forecasting
- **PuLP**: MILP optimization
- **NumPy/Pandas**: Data processing
- **Plotly/Dash**: Visualization

---

## Usage

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
├── main.py                   # CLI entry point (legacy)
├── main_multiagent.py       # Multi-agent CLI
├── calculate_baseline.py    # Baseline metrics
├── validate_junction2025.py # Validation suite
├── test_api.py              # API test suite
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image
├── docker-compose.yml       # Multi-service setup
├── .env.example             # Environment template
└── README.md                # This file
```

---

## Deployment

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

## Authors & Contact

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

