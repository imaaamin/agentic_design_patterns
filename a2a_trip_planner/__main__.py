"""
A2A Trip Planner - Main Entry Point

This module allows running the client via:
    python -m a2a_trip_planner

For starting individual agents, use:
    python -m a2a_trip_planner.agents.googleadk_hotels.hotel_agent
    python -m a2a_trip_planner.agents.crewai_flights.flight_agent
    python -m a2a_trip_planner.agents.langgraph_orchestrator.orchestrator
"""

from .run import main

if __name__ == "__main__":
    main()
