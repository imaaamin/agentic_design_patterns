"""
A2A Agents Package

Contains A2A-compliant agents built with different frameworks:
- langgraph_orchestrator: Main coordinator using LangGraph
- googleadk_hotels: Hotel search using Google ADK
- crewai_flights: Flight search using CrewAI

Each agent runs as a separate server and exposes:
- Agent Card at /.well-known/agent.json
- JSON-RPC 2.0 endpoint at /
"""

# Note: Don't import agents here to avoid circular imports
# Import them directly when needed:
#   from a2a_trip_planner.agents.googleadk_hotels import HotelAgentExecutor
#   from a2a_trip_planner.agents.crewai_flights import FlightAgentExecutor
#   from a2a_trip_planner.agents.langgraph_orchestrator import OrchestratorExecutor
