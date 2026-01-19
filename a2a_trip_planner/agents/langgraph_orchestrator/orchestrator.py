"""
LangGraph-based Trip Planner Orchestrator - A2A Protocol Compliant

This is the main entry point for the A2A Trip Planner system.
The orchestrator:
1. Receives trip requests from clients via A2A protocol
2. Discovers specialist agents by fetching their Agent Cards
3. Delegates tasks to Hotel (GoogleADK) and Flight (CrewAI) agents
4. Combines results into comprehensive trip recommendations

Based on: https://github.com/a2aproject/a2a-samples
Uses A2A SDK for protocol compliance and LangGraph for workflow orchestration.
"""

import os
import json
import logging
import asyncio
from typing import TypedDict, Annotated, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
import httpx

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

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
AGENT_HOST = "0.0.0.0"
AGENT_PORT = 8001

# URLs of specialist agents to discover
SPECIALIST_AGENTS = {
    "hotels": "http://localhost:8002",
    "flights": "http://localhost:8003",
}


# =============================================================================
# A2A Discovery and Communication
# =============================================================================

class AgentDiscovery:
    """
    Handles A2A agent discovery by fetching Agent Cards.
    
    Per A2A spec, agents expose their cards at /.well-known/agent.json
    """
    
    def __init__(self):
        self.discovered_agents: dict[str, dict] = {}
    
    async def discover_agent(self, name: str, base_url: str) -> dict:
        """
        Discover an agent by fetching its Agent Card.
        
        Args:
            name: Identifier for this agent (e.g., "hotels", "flights")
            base_url: Base URL of the agent
        
        Returns:
            The agent's Agent Card
        """
        card_url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        logger.info(f"🔍 Discovering agent '{name}' at {card_url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(card_url, timeout=10)
            response.raise_for_status()
            card = response.json()
        
        self.discovered_agents[name] = {
            "card": card,
            "url": base_url,
        }
        
        logger.info(f"   ✅ Discovered: {card.get('name', 'Unknown')}")
        logger.info(f"   Skills: {[s.get('id') for s in card.get('skills', [])]}")
        
        return card
    
    async def discover_all(self, agents: dict[str, str]) -> dict:
        """
        Discover multiple agents.
        
        Args:
            agents: Dict mapping name to base URL
        
        Returns:
            Dict of discovered agent info
        """
        for name, url in agents.items():
            try:
                await self.discover_agent(name, url)
            except Exception as e:
                logger.error(f"   ❌ Failed to discover {name}: {e}")
        
        return self.discovered_agents
    
    async def send_task(self, agent_name: str, message: str, data: dict = None) -> dict:
        """
        Send a task to a discovered agent using JSON-RPC 2.0.
        
        Args:
            agent_name: Name of the agent (e.g., "hotels")
            message: Text message to send
            data: Optional structured data
        
        Returns:
            Task result from the agent
        """
        if agent_name not in self.discovered_agents:
            raise ValueError(f"Agent '{agent_name}' not discovered")
        
        agent_url = self.discovered_agents[agent_name]["url"]
        
        import uuid
        message_id = str(uuid.uuid4())
        
        parts = [{"kind": "text", "text": message}]
        if data:
            parts.append({"kind": "data", "data": data})
        
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": parts
                },
                "configuration": {
                    "acceptedOutputModes": ["text"]
                }
            }
        }
        
        logger.info(f"📤 Sending task to {agent_name} at {agent_url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            logger.error(f"   ❌ Error from {agent_name}: {result['error']}")
            raise RuntimeError(f"A2A Error: {result['error']}")
        
        logger.info(f"   ✅ Task completed by {agent_name}")
        return result.get("result", {})


# =============================================================================
# LangGraph State and Workflow
# =============================================================================

class TripPlannerState(TypedDict):
    """State for the LangGraph trip planning workflow."""
    # Request info
    user_query: str
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str]
    
    # Agent results
    hotels_result: Optional[str]
    flights_result: Optional[str]
    
    # Final output
    recommendation: str
    status: str
    
    # Message history
    messages: Annotated[list, add_messages]


class TripPlannerWorkflow:
    """
    LangGraph workflow for trip planning.
    
    Orchestrates:
    1. Discovery of specialist agents
    2. Delegation to hotel agent
    3. Delegation to flight agent
    4. Synthesis of final recommendation
    """
    
    def __init__(self, discovery: AgentDiscovery):
        self.discovery = discovery
        self.llm = self._get_llm()
        self.workflow = self._build_workflow()
    
    def _get_llm(self):
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            if not os.getenv("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = google_key
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
        
        raise ValueError("Set GOOGLE_API_KEY or GROQ_API_KEY")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(TripPlannerState)
        
        # Add nodes
        workflow.add_node("discover_agents", self._discover_agents_node)
        workflow.add_node("search_flights", self._search_flights_node)
        workflow.add_node("search_hotels", self._search_hotels_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Set flow
        workflow.set_entry_point("discover_agents")
        workflow.add_edge("discover_agents", "search_flights")
        workflow.add_edge("search_flights", "search_hotels")
        workflow.add_edge("search_hotels", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    async def _discover_agents_node(self, state: TripPlannerState) -> dict:
        """Node: Discover specialist agents via A2A Agent Cards."""
        logger.info("\n" + "=" * 50)
        logger.info("🔍 STEP 1: Discovering specialist agents...")
        
        await self.discovery.discover_all(SPECIALIST_AGENTS)
        
        discovered = list(self.discovery.discovered_agents.keys())
        logger.info(f"   Discovered: {discovered}")
        
        return {
            "status": "agents_discovered",
            "messages": [AIMessage(content=f"Discovered agents: {discovered}")]
        }
    
    async def _search_flights_node(self, state: TripPlannerState) -> dict:
        """Node: Delegate flight search to CrewAI agent."""
        logger.info("\n" + "=" * 50)
        logger.info("✈️ STEP 2: Searching flights via CrewAI agent...")
        
        if "flights" not in self.discovery.discovered_agents:
            return {"flights_result": "Flight agent not available", "status": "flights_skipped"}
        
        query = f"""Find flights for this trip:
        - From: {state.get('origin', 'Unknown')}
        - To: {state.get('destination', 'Unknown')}
        - Departure: {state.get('departure_date', 'Unknown')}
        - Return: {state.get('return_date', 'One-way')}
        
        Original request: {state.get('user_query', '')}"""
        
        try:
            result = await self.discovery.send_task("flights", query)
            # Extract text from artifacts
            flights_text = self._extract_text_from_result(result)
            return {"flights_result": flights_text, "status": "flights_complete"}
        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            return {"flights_result": f"Error: {e}", "status": "flights_error"}
    
    async def _search_hotels_node(self, state: TripPlannerState) -> dict:
        """Node: Delegate hotel search to GoogleADK agent."""
        logger.info("\n" + "=" * 50)
        logger.info("🏨 STEP 3: Searching hotels via GoogleADK agent...")
        
        if "hotels" not in self.discovery.discovered_agents:
            return {"hotels_result": "Hotel agent not available", "status": "hotels_skipped"}
        
        query = f"""Find hotels for this trip:
        - Location: {state.get('destination', 'Unknown')}
        - Check-in: {state.get('departure_date', 'Unknown')}
        - Check-out: {state.get('return_date', 'Unknown')}
        
        Original request: {state.get('user_query', '')}"""
        
        try:
            result = await self.discovery.send_task("hotels", query)
            hotels_text = self._extract_text_from_result(result)
            return {"hotels_result": hotels_text, "status": "hotels_complete"}
        except Exception as e:
            logger.error(f"Hotel search failed: {e}")
            return {"hotels_result": f"Error: {e}", "status": "hotels_error"}
    
    async def _synthesize_node(self, state: TripPlannerState) -> dict:
        """Node: Synthesize final recommendation using LLM."""
        logger.info("\n" + "=" * 50)
        logger.info("📋 STEP 4: Synthesizing trip recommendation...")
        
        prompt = f"""You are a travel advisor. Create a comprehensive trip recommendation based on:

**TRIP REQUEST:**
{state.get('user_query', '')}

- From: {state.get('origin', 'Unknown')}
- To: {state.get('destination', 'Unknown')}
- Dates: {state.get('departure_date', '')} to {state.get('return_date', '')}

**FLIGHT OPTIONS:**
{state.get('flights_result', 'No flight data')}

**HOTEL OPTIONS:**
{state.get('hotels_result', 'No hotel data')}

Create a recommendation with:
1. **Best Flight** - Your top pick with reasoning
2. **Best Hotel** - Your top pick with reasoning
3. **Estimated Total Cost**
4. **Travel Tips** for the destination

Be enthusiastic and helpful!"""
        
        messages = [
            SystemMessage(content="You are an expert travel advisor creating personalized trip recommendations."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "recommendation": response.content,
            "status": "complete"
        }
    
    def _extract_text_from_result(self, result: dict) -> str:
        """Extract text content from A2A task result."""
        # Try to get from artifacts
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            parts = artifact.get("parts", [])
            for part in parts:
                if part.get("type") == "text" or "text" in part:
                    return part.get("text", str(part))
        
        # Try status message
        status = result.get("status", {})
        message = status.get("message", {})
        parts = message.get("parts", [])
        for part in parts:
            if "text" in part:
                return part["text"]
        
        # Fallback
        return str(result)
    
    async def run(self, user_query: str, origin: str, destination: str, 
                  departure_date: str, return_date: str = None) -> dict:
        """Run the trip planning workflow."""
        initial_state: TripPlannerState = {
            "user_query": user_query,
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "hotels_result": None,
            "flights_result": None,
            "recommendation": "",
            "status": "started",
            "messages": [HumanMessage(content=user_query)]
        }
        
        # Run async workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return final_state


# =============================================================================
# A2A Agent Executor
# =============================================================================

class OrchestratorExecutor(AgentExecutor):
    """
    A2A AgentExecutor for the LangGraph Trip Planner Orchestrator.
    
    This is the main entry point for client requests.
    It discovers specialist agents and coordinates the workflow.
    """
    
    def __init__(self):
        self.discovery = AgentDiscovery()
        self.workflow = TripPlannerWorkflow(self.discovery)
        logger.info("🌍 OrchestratorExecutor initialized")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a trip planning task."""
        
        query = context.get_user_input()
        logger.info(f"🌍 Processing trip request: {query[:100]}...")
        
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Update status
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Planning your trip... Discovering specialist agents.",
                    task.context_id,
                    task.id,
                ),
            )
            
            # Parse query to extract trip details (simple extraction)
            trip_info = self._parse_trip_query(query)
            
            # Run the LangGraph workflow
            result = await self.workflow.run(
                user_query=query,
                origin=trip_info.get("origin", "Unknown"),
                destination=trip_info.get("destination", "Unknown"),
                departure_date=trip_info.get("departure_date", "2026-06-15"),
                return_date=trip_info.get("return_date", "2026-06-22"),
            )
            
            # Add recommendation as artifact
            await updater.add_artifact(
                [Part(root=TextPart(text=result.get("recommendation", "No recommendation")))],
                name='trip_recommendation',
            )
            
            await updater.complete()
            logger.info(f"🌍 Trip planning completed for task {task.id}")
            
        except Exception as e:
            logger.error(f"🌍 Trip planning error: {e}")
            import traceback
            traceback.print_exc()
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"Trip planning failed: {str(e)}",
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
    
    def _parse_trip_query(self, query: str) -> dict:
        """Simple parser to extract trip details from query."""
        query_lower = query.lower()
        
        # Default values
        result = {
            "origin": "New York",
            "destination": "Paris",
            "departure_date": "2026-06-15",
            "return_date": "2026-06-22",
        }
        
        # Try to find cities
        cities = ["paris", "london", "tokyo", "new york", "nyc", "los angeles"]
        found_cities = [c for c in cities if c in query_lower]
        
        if len(found_cities) >= 2:
            result["origin"] = found_cities[0].title()
            result["destination"] = found_cities[1].title()
        elif len(found_cities) == 1:
            result["destination"] = found_cities[0].title()
        
        return result
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel a running task."""
        logger.info("🌍 Task cancellation requested")
        raise ServerError(error=UnsupportedOperationError())


# =============================================================================
# Agent Card Definition
# =============================================================================

def get_agent_card() -> AgentCard:
    """Create the A2A Agent Card for the orchestrator."""
    return AgentCard(
        name="Trip Planner Orchestrator (LangGraph)",
        description="Client-facing trip planning coordinator. Discovers and coordinates specialist agents (flights, hotels) to provide comprehensive trip recommendations.",
        url=f"http://localhost:{AGENT_PORT}",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        skills=[
            AgentSkill(
                id="plan_trip",
                name="Trip Planning",
                description="Plan a complete trip by coordinating with specialist agents for flights and hotels.",
                tags=["travel", "planning", "orchestration"],
                examples=[
                    "Plan a trip from New York to Paris from June 15-22, 2026",
                    "I want to visit Tokyo next month with a budget of $2000",
                ],
            )
        ],
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the A2A Trip Planner Orchestrator Server."""
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🌍 Starting Trip Planner Orchestrator (A2A Protocol)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This orchestrator will discover these specialist agents:")
    for name, url in SPECIALIST_AGENTS.items():
        logger.info(f"   - {name}: {url}")
    logger.info("")
    logger.info("Make sure to start the specialist agents first!")
    logger.info("")
    
    agent_card = get_agent_card()
    
    # Use DefaultRequestHandler as shown in official A2A samples
    request_handler = DefaultRequestHandler(
        agent_executor=OrchestratorExecutor(),
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
