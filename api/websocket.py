"""
WebSocket connection manager for live updates.
Supports both legacy LiveUpdate format and new frontend LiveSystemUpdate format.
"""

import asyncio
import json
from typing import List, Set, Optional, TYPE_CHECKING
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from api.models import LiveUpdate

if TYPE_CHECKING:
    from api.services import IFOService


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task = None
        self.ifo_service: Optional['IFOService'] = None
    
    def set_service(self, service: 'IFOService'):
        """Set the IFO service instance."""
        self.ifo_service = service
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_status_updates(self, interval: int = 2):
        """Continuously broadcast system status updates."""
        while True:
            try:
                if self.active_connections and self.ifo_service:
                    # Get current status
                    status = self.ifo_service.get_system_status()
                    
                    # Create live update message (legacy format)
                    update = LiveUpdate(
                        timestamp=datetime.now(),
                        pumps=status.pumps,
                        tunnel=status.tunnel,
                        total_power=status.total_power,
                        event_type="status_update"
                    )
                    
                    # Broadcast to all clients
                    message = update.model_dump(mode='json')
                    
                    # Add frontend-specific fields
                    message['type'] = 'system_update'
                    message['dashboard'] = {
                        'currentPower': status.total_power,
                        'tunnelLevel': status.tunnel.level,
                        'tunnelVolume': status.tunnel.volume,
                        'inflowRate': status.tunnel.inflow_rate,
                        'outflowRate': status.tunnel.outflow_rate,
                        'activePumps': sum(1 for p in status.pumps if p.is_running),
                    }
                    
                    await self.broadcast(message)
                
                # Wait before next update
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Error in broadcast loop: {e}")
                await asyncio.sleep(interval)
    
    def start_broadcasting(self, interval: int = 2):
        """Start background broadcasting task."""
        if self.broadcast_task is None or self.broadcast_task.done():
            self.broadcast_task = asyncio.create_task(
                self.broadcast_status_updates(interval)
            )
    
    def stop_broadcasting(self):
        """Stop background broadcasting task."""
        if self.broadcast_task and not self.broadcast_task.done():
            self.broadcast_task.cancel()


# Singleton connection manager
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get WebSocket connection manager."""
    return manager
