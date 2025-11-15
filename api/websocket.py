"""
WebSocket connection manager for live updates.
"""

import asyncio
import json
from typing import List, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from api.models import LiveUpdate
from api.services import get_ifo_service


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task = None
    
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
        ifo_service = get_ifo_service()
        
        while True:
            try:
                if self.active_connections:
                    # Get current status
                    status = ifo_service.get_system_status()
                    
                    # Create live update message
                    update = LiveUpdate(
                        timestamp=datetime.now(),
                        pumps=status.pumps,
                        tunnel=status.tunnel,
                        total_power=status.total_power,
                        event_type="status_update"
                    )
                    
                    # Broadcast to all clients
                    await self.broadcast(update.model_dump(mode='json'))
                
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
