"""CrewAI-based Flight Search Agent - A2A Protocol

Run this agent with:
    uv run python -m a2a_trip_planner.agents.crewai_flights.flight_agent
"""

# Lazy imports to avoid issues when a2a-sdk isn't installed
def get_executor():
    from .flight_agent import FlightAgentExecutor
    return FlightAgentExecutor

def get_agent_card():
    from .flight_agent import get_agent_card as _get_card
    return _get_card()
