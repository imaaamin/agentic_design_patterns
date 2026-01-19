"""Google ADK-based Hotel Search Agent - A2A Protocol

Run this agent with:
    uv run python -m a2a_trip_planner.agents.googleadk_hotels.hotel_agent
"""

# Lazy imports to avoid issues when a2a-sdk isn't installed
def get_executor():
    from .hotel_agent import HotelAgentExecutor
    return HotelAgentExecutor

def get_agent_card():
    from .hotel_agent import get_agent_card as _get_card
    return _get_card()
