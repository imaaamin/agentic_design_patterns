"""
CrewAI-based Flight Search Agent - A2A Protocol Compliant

This agent implements the A2A protocol using the official a2a-sdk pattern.
It exposes flight search capabilities via an AgentCard and handles tasks
via JSON-RPC 2.0.

Based on: https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent_executor.py

A2A Protocol:
- Agent Card served at /.well-known/agent.json
- JSON-RPC 2.0 endpoint for message/send
- Uses CrewAI internally for flight search logic
"""

import os
import json
import logging
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Any

from dotenv import load_dotenv

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    Part,
    TextPart,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

# CrewAI imports
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
AGENT_HOST = "0.0.0.0"
AGENT_PORT = 8003


# =============================================================================
# Mock Flight Data
# =============================================================================

@dataclass
class FlightResult:
    airline: str
    flight_number: str
    departure_airport: str
    arrival_airport: str
    departure_time: str
    arrival_time: str
    duration_hours: float
    price: float
    stops: int
    cabin_class: str


MOCK_FLIGHTS = {
    ("NYC", "PAR"): [
        FlightResult("Air France", "AF007", "JFK", "CDG", "08:00", "20:30", 7.5, 650, 0, "Economy"),
        FlightResult("Delta", "DL123", "JFK", "CDG", "18:00", "08:15", 7.25, 720, 0, "Economy"),
        FlightResult("United", "UA456", "EWR", "CDG", "10:30", "00:45", 8.25, 580, 1, "Economy"),
        FlightResult("Norwegian", "DY7001", "JFK", "CDG", "14:00", "03:30", 7.5, 450, 0, "Economy"),
    ],
    ("NYC", "LON"): [
        FlightResult("British Airways", "BA178", "JFK", "LHR", "19:00", "07:00", 7, 780, 0, "Economy"),
        FlightResult("Virgin Atlantic", "VS10", "JFK", "LHR", "21:30", "09:25", 6.9, 650, 0, "Economy"),
    ],
    ("NYC", "TYO"): [
        FlightResult("Japan Airlines", "JL003", "JFK", "NRT", "12:00", "15:00+1", 14, 1200, 0, "Economy"),
        FlightResult("ANA", "NH9", "JFK", "HND", "11:30", "15:30+1", 14, 1350, 0, "Economy"),
    ],
}

CITY_CODES = {
    "new york": "NYC", "nyc": "NYC", "paris": "PAR",
    "london": "LON", "tokyo": "TYO", "los angeles": "LAX",
}


def normalize_city(city: str) -> str:
    return CITY_CODES.get(city.lower().strip(), city.upper()[:3])


# =============================================================================
# CrewAI Tool for Flight Search
# =============================================================================

@tool("Flight Search Tool")
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    max_price: Optional[float] = None,
    max_stops: Optional[int] = None
) -> str:
    """Search for flights between two cities."""
    origin_code = normalize_city(origin)
    dest_code = normalize_city(destination)
    
    flights = MOCK_FLIGHTS.get((origin_code, dest_code), [])
    
    if not flights:
        return f"No flights found from {origin} to {destination}."
    
    filtered = flights.copy()
    if max_price:
        filtered = [f for f in filtered if f.price <= max_price]
    if max_stops is not None:
        filtered = [f for f in filtered if f.stops <= max_stops]
    
    if not filtered:
        return f"No flights match criteria. {len(flights)} available without filters."
    
    filtered.sort(key=lambda x: x.price)
    
    results = [f"Found {len(filtered)} flights {origin_code} → {dest_code}:\n"]
    for i, f in enumerate(filtered[:5], 1):
        stops = "Direct" if f.stops == 0 else f"{f.stops} stop(s)"
        results.append(
            f"{i}. {f.airline} {f.flight_number} | "
            f"{f.departure_time}-{f.arrival_time} | "
            f"{f.duration_hours}h | {stops} | ${f.price}\n"
        )
    return "".join(results)


# =============================================================================
# CrewAI Agent Setup
# =============================================================================

class FlightSearchCrewAgent:
    """CrewAI-based flight search logic."""
    
    def __init__(self):
        self.llm = self._get_llm()
        self.agent = Agent(
            role='Flight Search Specialist',
            goal='Find the best flight options for travelers',
            backstory="Expert travel agent with airline and pricing knowledge.",
            verbose=False,
            tools=[search_flights],
            allow_delegation=False,
            llm=self.llm,
        )
    
    def _get_llm(self) -> LLM:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_key)
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            return LLM(model="gemini/gemini-2.0-flash", api_key=google_key)
        raise ValueError("No API key found. Set GROQ_API_KEY or GOOGLE_API_KEY")
    
    def search(self, query: str) -> str:
        """Execute a flight search query."""
        task = Task(
            description=f"Search for flights based on: {query}\nUse the Flight Search Tool.",
            expected_output="Flight recommendations with details",
            agent=self.agent
        )
        crew = Crew(agents=[self.agent], tasks=[task], verbose=False)
        return str(crew.kickoff())


# =============================================================================
# A2A Agent Executor (following official SDK pattern)
# =============================================================================

class FlightAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor for the CrewAI Flight Search Agent.
    
    This follows the pattern from the official a2a-samples:
    https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent_executor.py
    """
    
    def __init__(self):
        self.crew_agent = FlightSearchCrewAgent()
        logger.info("✈️ FlightAgentExecutor initialized")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a flight search task."""
        
        query = context.get_user_input()
        logger.info(f"✈️ Processing flight search: {query[:100]}...")
        
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Update status to working
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Searching for flights...",
                    task.context_id,
                    task.id,
                ),
            )
            
            # Execute the CrewAI agent
            result = self.crew_agent.search(query)
            
            # Add result as artifact
            await updater.add_artifact(
                [Part(root=TextPart(text=result))],
                name='flight_search_result',
            )
            
            # Mark as complete
            await updater.complete()
            logger.info(f"✈️ Flight search completed for task {task.id}")
            
        except Exception as e:
            logger.error(f"✈️ Flight search error: {e}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"Flight search failed: {str(e)}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel a running task."""
        logger.info("✈️ Task cancellation requested")
        raise ServerError(error=UnsupportedOperationError())


# =============================================================================
# Agent Card Definition
# =============================================================================

def get_agent_card() -> AgentCard:
    """Create the A2A Agent Card for this agent."""
    return AgentCard(
        name="CrewAI Flight Search Agent",
        description="Finds the best flight options for trips using CrewAI. Searches routes, filters by price and stops, and provides recommendations.",
        url=f"http://localhost:{AGENT_PORT}",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=False,
        ),
        skills=[
            AgentSkill(
                id="find_flights",
                name="Flight Search",
                description="Search for flights between cities with filters for budget, stops, and dates.",
                tags=["travel", "flights", "booking"],
                examples=[
                    "Find flights from New York to Paris on June 15, 2026",
                    "Search for non-stop flights to London under $800",
                ],
            )
        ],
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the A2A Flight Agent Server."""
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("✈️ Starting CrewAI Flight Agent (A2A Protocol)")
    logger.info("=" * 60)
    
    agent_card = get_agent_card()
    
    # Use DefaultRequestHandler as shown in official A2A samples
    request_handler = DefaultRequestHandler(
        agent_executor=FlightAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    logger.info(f"📍 Agent Card: http://localhost:{AGENT_PORT}/.well-known/agent.json")
    logger.info(f"📍 JSON-RPC: http://localhost:{AGENT_PORT}/")
    
    uvicorn.run(app.build(), host=AGENT_HOST, port=AGENT_PORT)


if __name__ == "__main__":
    main()
