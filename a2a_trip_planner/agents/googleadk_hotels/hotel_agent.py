"""
Google ADK-based Hotel Search Agent - A2A Protocol Compliant

This agent implements the A2A protocol using the official a2a-sdk pattern.
It exposes hotel search capabilities via an AgentCard and handles tasks
via JSON-RPC 2.0.

Based on: https://github.com/a2aproject/a2a-samples

A2A Protocol:
- Agent Card served at /.well-known/agent.json
- JSON-RPC 2.0 endpoint for message/send
- Uses Google ADK internally for hotel search logic
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import nest_asyncio

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

# Google ADK imports
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
AGENT_HOST = "0.0.0.0"
AGENT_PORT = 8002


# =============================================================================
# Mock Hotel Data
# =============================================================================

@dataclass
class HotelResult:
    name: str
    location: str
    price_per_night: float
    rating: float
    review_count: int
    amenities: list
    distance_to_center_km: float


MOCK_HOTELS = {
    "PAR": [
        HotelResult("Le Marais Boutique", "Le Marais, Paris", 185, 4.7, 1240,
                   ["WiFi", "Breakfast", "AC", "Minibar"], 0.5),
        HotelResult("Hotel Saint-Germain", "Saint-Germain-des-Prés", 220, 4.8, 890,
                   ["WiFi", "Breakfast", "Spa", "Restaurant"], 1.2),
        HotelResult("Montmartre Inn", "Montmartre, Paris", 120, 4.3, 2100,
                   ["WiFi", "AC", "24h Reception"], 2.5),
        HotelResult("Bastille Budget Stay", "Bastille, Paris", 85, 4.0, 3200,
                   ["WiFi", "AC"], 1.8),
    ],
    "LON": [
        HotelResult("Kensington Gardens Hotel", "Kensington, London", 195, 4.6, 1560,
                   ["WiFi", "Breakfast", "AC", "Bar"], 0.8),
        HotelResult("Westminster Grand", "Westminster, London", 320, 4.8, 720,
                   ["WiFi", "Breakfast", "Spa", "Restaurant", "Gym"], 0.2),
    ],
    "TYO": [
        HotelResult("Shibuya Sky Hotel", "Shibuya, Tokyo", 180, 4.7, 2340,
                   ["WiFi", "Onsen", "Breakfast", "AC"], 0.5),
        HotelResult("Shinjuku Business Hotel", "Shinjuku, Tokyo", 95, 4.3, 4500,
                   ["WiFi", "AC", "24h Convenience"], 1.0),
    ],
}

CITY_CODES = {
    "paris": "PAR", "london": "LON", "tokyo": "TYO",
    "new york": "NYC", "los angeles": "LAX",
}


def normalize_city(city: str) -> str:
    return CITY_CODES.get(city.lower().strip(), city.upper()[:3])


# =============================================================================
# Google ADK Tool for Hotel Search
# =============================================================================

def search_hotels_tool(
    location: str,
    check_in_date: str,
    check_out_date: str,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None
) -> str:
    """
    Search for hotels in a given location.
    
    Args:
        location: City name or code
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        max_price: Maximum price per night
        min_rating: Minimum star rating
    """
    city_code = normalize_city(location)
    hotels = MOCK_HOTELS.get(city_code, [])
    
    if not hotels:
        return f"No hotels found in {location}. Try Paris, London, or Tokyo."
    
    filtered = hotels.copy()
    if max_price:
        filtered = [h for h in filtered if h.price_per_night <= max_price]
    if min_rating:
        filtered = [h for h in filtered if h.rating >= min_rating]
    
    if not filtered:
        return f"No hotels match criteria. {len(hotels)} available without filters."
    
    filtered.sort(key=lambda x: (-x.rating, x.price_per_night))
    
    # Calculate nights
    from datetime import datetime
    try:
        d1 = datetime.strptime(check_in_date, "%Y-%m-%d")
        d2 = datetime.strptime(check_out_date, "%Y-%m-%d")
        nights = (d2 - d1).days
    except:
        nights = 1
    
    results = [f"Found {len(filtered)} hotels in {location} ({nights} nights):\n"]
    for i, h in enumerate(filtered[:5], 1):
        total = h.price_per_night * nights
        amenities = ", ".join(h.amenities[:3])
        results.append(
            f"{i}. {h.name} ⭐{h.rating}\n"
            f"   📍 {h.location} ({h.distance_to_center_km}km from center)\n"
            f"   💰 ${h.price_per_night}/night (Total: ${total})\n"
            f"   ✨ {amenities}\n"
        )
    return "".join(results)


# =============================================================================
# Google ADK Agent Setup
# =============================================================================

class HotelSearchADKAgent:
    """Google ADK-based hotel search logic."""
    
    APP_NAME = "hotel_search_agent"
    USER_ID = "a2a_user"
    
    def __init__(self):
        self._setup_api_key()
        self.agent = Agent(
            name="hotel_search_specialist",
            model="gemini-2.0-flash",
            description="Expert hotel search agent that finds the best accommodations.",
            instruction="""You are a professional hotel concierge.
            Use the search_hotels function to find hotels.
            Provide recommendations with reasoning about location, value, and amenities.""",
            tools=[search_hotels_tool]
        )
        logger.info("🏨 HotelSearchADKAgent initialized")
    
    def _setup_api_key(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = api_key
    
    async def search(self, query: str) -> str:
        """Execute a hotel search query."""
        session_service = InMemorySessionService()
        session_id = f"session_{id(self)}"
        
        session = await session_service.create_session(
            app_name=self.APP_NAME,
            user_id=self.USER_ID,
            session_id=session_id
        )
        
        runner = Runner(
            agent=self.agent,
            app_name=self.APP_NAME,
            session_service=session_service
        )
        
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )
        
        final_response = ""
        async for event in runner.run_async(
            user_id=self.USER_ID,
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response():
                final_response = event.content.parts[0].text
        
        return final_response


# =============================================================================
# A2A Agent Executor (following official SDK pattern)
# =============================================================================

class HotelAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor for the Google ADK Hotel Search Agent.
    
    Based on the official a2a-samples pattern.
    """
    
    def __init__(self):
        self.adk_agent = HotelSearchADKAgent()
        logger.info("🏨 HotelAgentExecutor initialized")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a hotel search task."""
        
        query = context.get_user_input()
        logger.info(f"🏨 Processing hotel search: {query[:100]}...")
        
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Searching for hotels...",
                    task.context_id,
                    task.id,
                ),
            )
            
            # Execute Google ADK agent
            result = await self.adk_agent.search(query)
            
            await updater.add_artifact(
                [Part(root=TextPart(text=result))],
                name='hotel_search_result',
            )
            
            await updater.complete()
            logger.info(f"🏨 Hotel search completed for task {task.id}")
            
        except Exception as e:
            logger.error(f"🏨 Hotel search error: {e}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f"Hotel search failed: {str(e)}",
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
        logger.info("🏨 Task cancellation requested")
        raise ServerError(error=UnsupportedOperationError())


# =============================================================================
# Agent Card Definition
# =============================================================================

def get_agent_card() -> AgentCard:
    """Create the A2A Agent Card for this agent."""
    return AgentCard(
        name="Google ADK Hotel Search Agent",
        description="Finds the best hotel accommodations using Google ADK. Searches by location, filters by budget and rating, and provides recommendations.",
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
                id="find_hotels",
                name="Hotel Search",
                description="Search for hotels in a location with filters for budget, rating, and amenities.",
                tags=["travel", "hotels", "accommodation"],
                examples=[
                    "Find hotels in Paris from June 15-22, 2026",
                    "Search for 4-star hotels with WiFi under $200/night",
                ],
            )
        ],
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the A2A Hotel Agent Server."""
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🏨 Starting Google ADK Hotel Agent (A2A Protocol)")
    logger.info("=" * 60)
    
    agent_card = get_agent_card()
    
    # Use DefaultRequestHandler as shown in official A2A samples
    request_handler = DefaultRequestHandler(
        agent_executor=HotelAgentExecutor(),
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
