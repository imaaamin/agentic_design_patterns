"""LangGraph-based Trip Planner Orchestrator - A2A Protocol

Run this agent with:
    uv run python -m a2a_trip_planner.agents.langgraph_orchestrator.orchestrator
"""

# Lazy imports to avoid issues when a2a-sdk isn't installed
def get_executor():
    from .orchestrator import OrchestratorExecutor
    return OrchestratorExecutor

def get_agent_card():
    from .orchestrator import get_agent_card as _get_card
    return _get_card()
