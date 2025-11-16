"""
AquaBot: Gemini-powered chatbot for IFO system assistance.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

from .config import settings

log = logging.getLogger("aquabot")


class AquaBot:
    """
    Chatbot powered by Google Gemini API for IFO system assistance.
    Provides context-aware responses about pump optimization, system status, and best practices.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AquaBot with Gemini API.
        
        Args:
            api_key: Google API key. If None, reads from settings.GEMINI_API_KEY.
        """
        if not GEMINI_OK:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or config")
        
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash for fast, efficient responses
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # System context about IFO
        self.system_context = """You are AquaBot, an AI assistant for the IFO (Intelligent Flood Optimization) system.

IFO is a multi-agent wastewater pump optimization system that uses:
- Model Predictive Control (MPC) with MILP optimization
- LSTM forecasting for inflow predictions
- Discrete pump frequencies: 48, 49, 50 Hz
- Affinity laws for pump modeling (Q∝f, H∝f², P∝f³)
- 96-step optimization horizon (24 hours with 15-minute intervals)

Your role:
1. Answer questions about pump optimization, system status, and best practices
2. Explain KPIs, energy savings, and operational constraints
3. Help troubleshoot issues and interpret system data
4. Provide guidance on configuration and deployment
5. Be concise and technical but accessible

Key capabilities of the IFO system:
- Real-time pump control via OPC UA
- REST API + WebSocket for monitoring
- Docker deployment and ARM/edge support
- Typical 10-30% energy cost savings
- Zero constraint violations with safe operation

When discussing data or status, remind users to check the API endpoints or dashboard for real-time values.
Keep responses brief unless detailed explanation is requested."""
        
        self.conversation_history: List[Dict[str, str]] = []
        log.info("AquaBot initialized with Gemini 1.5 Flash")

    def chat(self, user_message: str, system_status: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user message and generate response.
        
        Args:
            user_message: User's question or message
            system_status: Optional current system status for context-aware responses
            
        Returns:
            AquaBot's response
        """
        # Build context with system status if provided
        context_parts = [self.system_context]
        
        if system_status:
            context_parts.append("\n\nCurrent System Status:")
            context_parts.append(f"- Total Power: {system_status.get('total_power', 'N/A')} kW")
            context_parts.append(f"- Tunnel Volume: {system_status.get('tunnel', {}).get('volume', 'N/A')} m³")
            context_parts.append(f"- Optimization Active: {system_status.get('optimization_active', False)}")
            
            pumps = system_status.get('pumps', [])
            if pumps:
                context_parts.append(f"- Active Pumps: {len([p for p in pumps if p.get('is_running')])} / {len(pumps)}")
        
        # Add conversation history for context (last 5 exchanges)
        messages = []
        for entry in self.conversation_history[-5:]:
            messages.append(f"User: {entry['user']}")
            messages.append(f"AquaBot: {entry['assistant']}")
        
        # Build final prompt
        full_context = "\n".join(context_parts)
        conversation = "\n".join(messages) if messages else ""
        
        prompt = f"{full_context}\n\n{conversation}\n\nUser: {user_message}\n\nAquaBot:"
        
        try:
            response = self.model.generate_content(prompt)
            assistant_reply = response.text.strip()
            
            # Store in history
            self.conversation_history.append({
                "user": user_message,
                "assistant": assistant_reply,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep history manageable (last 20 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_reply
            
        except Exception as e:
            log.error(f"Gemini API error: {e}")
            return f"I'm sorry, I encountered an error processing your request. Please try again. (Error: {str(e)})"

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        log.info("Conversation history reset")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()


# Singleton instance
_aquabot_instance: Optional[AquaBot] = None


def get_aquabot() -> AquaBot:
    """Get or create AquaBot singleton instance."""
    global _aquabot_instance
    if _aquabot_instance is None:
        try:
            _aquabot_instance = AquaBot()
        except Exception as e:
            log.error(f"Failed to initialize AquaBot: {e}")
            raise
    return _aquabot_instance
