"""
Test API endpoints to verify installation.
Run after starting the API server with: uvicorn api.main:app --reload
"""

import requests
import json
from datetime import datetime, timedelta


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("🔍 Testing /api/health...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    return response.status_code == 200


def test_status():
    """Test system status endpoint."""
    print("🔍 Testing /api/status...")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"   Status: {response.status_code}")
    if response.ok:
        data = response.json()
        print(f"   Pumps: {len(data['pumps'])}")
        print(f"   Total Power: {data['total_power']:.2f} kW")
        print(f"   Tunnel Volume: {data['tunnel']['volume']:.2f} m³\n")
    return response.status_code == 200


def test_pumps():
    """Test pumps endpoint."""
    print("🔍 Testing /api/pumps...")
    response = requests.get(f"{BASE_URL}/api/pumps")
    print(f"   Status: {response.status_code}")
    if response.ok:
        pumps = response.json()
        for pump in pumps:
            print(f"   Pump {pump['id']}: {pump['frequency']:.1f} Hz, {pump['flow']:.1f} m³/h, {pump['power']:.1f} kW")
    print()
    return response.status_code == 200


def test_tunnel():
    """Test tunnel metrics endpoint."""
    print("🔍 Testing /api/tunnel...")
    response = requests.get(f"{BASE_URL}/api/tunnel")
    print(f"   Status: {response.status_code}")
    if response.ok:
        tunnel = response.json()
        print(f"   Volume: {tunnel['volume']:.2f} m³")
        print(f"   Level: {tunnel['level']:.2f} m")
        print(f"   Inflow: {tunnel['inflow_rate']:.2f} m³/h")
        print(f"   Outflow: {tunnel['outflow_rate']:.2f} m³/h\n")
    return response.status_code == 200


def test_forecast():
    """Test forecast endpoint."""
    print("🔍 Testing /api/forecast...")
    response = requests.get(f"{BASE_URL}/api/forecast?horizon=24")
    print(f"   Status: {response.status_code}")
    if response.ok:
        forecast = response.json()
        print(f"   Horizon: {len(forecast['inflow_predictions'])} timesteps")
        print(f"   Model: {forecast['model_type']}")
        print(f"   First prediction: {forecast['inflow_predictions'][0]:.2f} m³/h\n")
    return response.status_code == 200


def test_optimize():
    """Test optimization endpoint."""
    print("🔍 Testing /api/optimize...")
    payload = {
        "horizon": 24,
        "mode": "offline"
    }
    response = requests.post(
        f"{BASE_URL}/api/optimize",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {response.status_code}")
    if response.ok:
        result = response.json()
        print(f"   Status: {result['status']}")
        print(f"   Total Energy: {result['total_energy']:.2f} kWh")
        print(f"   Computation Time: {result['computation_time']:.2f}s")
        if result.get('cost_savings'):
            print(f"   Cost Savings: {result['cost_savings']:.1f}%\n")
    return response.status_code == 200


def test_kpis():
    """Test KPIs endpoint."""
    print("🔍 Testing /api/kpis...")
    start = (datetime.now() - timedelta(days=1)).isoformat()
    end = datetime.now().isoformat()
    response = requests.get(f"{BASE_URL}/api/kpis?start={start}&end={end}")
    print(f"   Status: {response.status_code}")
    if response.ok:
        kpis = response.json()
        print(f"   Total Energy: {kpis['total_energy_consumed']:.2f} kWh")
        print(f"   Average Power: {kpis['average_power']:.2f} kW")
        print(f"   Cost Savings: {kpis['cost_savings_vs_baseline']:.1f}%\n")
    return response.status_code == 200


def main():
    """Run all API tests."""
    print("=" * 60)
    print("IFO Backend API Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Health Check", test_health),
        ("System Status", test_status),
        ("Pumps", test_pumps),
        ("Tunnel Metrics", test_tunnel),
        ("Forecast", test_forecast),
        ("Optimization", test_optimize),
        ("KPIs", test_kpis),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to API at {BASE_URL}")
            print("   Make sure the API is running: uvicorn api.main:app --reload\n")
            return
        except Exception as e:
            print(f"❌ Error in {name}: {e}\n")
            results[name] = False
    
    # Summary
    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
