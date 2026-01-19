"""
A2A (Agent-to-Agent) Trip Planner Example

This module demonstrates the official A2A Protocol with three agents:

- LangGraph Orchestrator: Client-facing coordinator that discovers and delegates
- Google ADK Hotels Agent: Specialist for hotel search
- CrewAI Flights Agent: Specialist for flight search

Based on: https://github.com/a2aproject/A2A
SDK: https://a2a-protocol.org/latest/sdk/python/

Usage:
    # Start agents in separate terminals:
    uv run python -m a2a_trip_planner.agents.googleadk_hotels.hotel_agent
    uv run python -m a2a_trip_planner.agents.crewai_flights.flight_agent
    uv run python -m a2a_trip_planner.agents.langgraph_orchestrator.orchestrator

    # Test with client:
    uv run python a2a_trip_planner/run.py --client
"""

__version__ = "1.0.0"
__a2a_version__ = "0.3.0"

# Agent ports for reference
ORCHESTRATOR_PORT = 8001
HOTEL_AGENT_PORT = 8002
FLIGHT_AGENT_PORT = 8003

AGENT_URLS = {
    "orchestrator": f"http://localhost:{ORCHESTRATOR_PORT}",
    "hotels": f"http://localhost:{HOTEL_AGENT_PORT}",
    "flights": f"http://localhost:{FLIGHT_AGENT_PORT}",
}

# Note: Don't import agents here to avoid import issues
# Import them directly when needed
