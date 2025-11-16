"""
Test script for frontend API endpoints.
Run this to verify all endpoints are working correctly.
"""

import requests
import json
from datetime import datetime

# Base URL
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"


def test_endpoint(method, endpoint, description, params=None, json_data=None):
    """Test a single endpoint."""
    url = f"{API_URL}{endpoint}"
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"{method} {url}")
    if params:
        print(f"Params: {params}")
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=5)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code == 200:
            print(f"✅ SUCCESS - Status: {response.status_code}")
            data = response.json()
            
            # Pretty print first few lines of response
            json_str = json.dumps(data, indent=2)
            lines = json_str.split('\n')
            preview = '\n'.join(lines[:20])
            if len(lines) > 20:
                preview += f"\n... ({len(lines) - 20} more lines)"
            print(f"Response preview:\n{preview}")
            return True
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(f"Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR - Is the server running on {BASE_URL}?")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT - Request took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False


def main():
    """Run all frontend API tests."""
    print("🚀 Frontend API Integration Test Suite")
    print(f"Testing backend at: {BASE_URL}")
    print(f"Current time: {datetime.now().isoformat()}")
    
    results = []
    
    # Test health check first
    print("\n" + "="*70)
    print("PREREQUISITE: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy and running")
            health = response.json()
            print(f"   Version: {health.get('version', 'N/A')}")
            print(f"   Service initialized: {health.get('service_initialized', False)}")
        else:
            print("❌ Backend health check failed")
            print("⚠️  Continuing tests anyway...")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("⚠️  Make sure the server is running with:")
        print("     python -m uvicorn api.main:app --reload")
        return
    
    # Dashboard endpoints
    results.append(test_endpoint(
        "GET", "/dashboard",
        "Dashboard - Complete dashboard data"
    ))
    
    # Forecast Agent
    results.append(test_endpoint(
        "GET", "/agents/forecast",
        "Forecast Agent - Default 24h horizon"
    ))
    
    results.append(test_endpoint(
        "GET", "/agents/forecast",
        "Forecast Agent - 48h horizon",
        params={"horizon": 48}
    ))
    
    # Planner Agent
    results.append(test_endpoint(
        "GET", "/agents/planner",
        "Planner Agent - Normal price scenario"
    ))
    
    results.append(test_endpoint(
        "GET", "/agents/planner",
        "Planner Agent - High price scenario",
        params={"price_scenario": "high"}
    ))
    
    # Executor Agent
    results.append(test_endpoint(
        "GET", "/agents/executor",
        "Executor Agent - Idle state"
    ))
    
    results.append(test_endpoint(
        "GET", "/agents/executor",
        "Executor Agent - Executing state",
        params={"is_executing": True}
    ))
    
    # Supervisor Agent
    results.append(test_endpoint(
        "GET", "/agents/supervisor",
        "Supervisor Agent - Aggregated metrics"
    ))
    
    # Simulations
    results.append(test_endpoint(
        "GET", "/simulations",
        "Simulations - Normal scenario"
    ))
    
    results.append(test_endpoint(
        "GET", "/simulations",
        "Simulations - Storm scenario",
        params={"scenario": "storm"}
    ))
    
    # Reports
    results.append(test_endpoint(
        "GET", "/reports",
        "Reports - Weekly data"
    ))
    
    results.append(test_endpoint(
        "GET", "/reports",
        "Reports - Monthly data",
        params={"time_range": "month"}
    ))
    
    # System Overview
    results.append(test_endpoint(
        "GET", "/system/overview",
        "System Overview - Agent architecture"
    ))
    
    # Settings
    results.append(test_endpoint(
        "GET", "/settings",
        "Settings - System configuration"
    ))
    
    results.append(test_endpoint(
        "POST", "/settings/opcua/test",
        "Settings - OPC UA connection test"
    ))
    
    # Pump Details
    results.append(test_endpoint(
        "GET", "/pumps/details",
        "Pump Details - Small pump H-Q curves"
    ))
    
    results.append(test_endpoint(
        "GET", "/pumps/details",
        "Pump Details - Large pump efficiency",
        params={"pump_type": "large", "curve_type": "efficiency"}
    ))
    
    # Notifications
    results.append(test_endpoint(
        "GET", "/notifications",
        "Notifications - Get all notifications"
    ))
    
    results.append(test_endpoint(
        "POST", "/notifications/1/read",
        "Notifications - Mark as read"
    ))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {total - passed} ❌")
    print(f"Success rate: {percentage:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The backend is ready for frontend integration.")
        print("\n📖 Next steps:")
        print("   1. Update frontend API_BASE_URL to http://localhost:8000")
        print("   2. Connect WebSocket to ws://localhost:8000/ws/live")
        print("   3. Start making requests from your React components")
        print("   4. Check FRONTEND_INTEGRATION.md for detailed integration guide")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("   Make sure all services are initialized correctly.")
    
    print("\n🌐 Useful links:")
    print(f"   API Docs: {BASE_URL}/api/docs")
    print(f"   Health: {BASE_URL}/api/health")
    print(f"   Frontend API: {API_URL}")


if __name__ == "__main__":
    main()
