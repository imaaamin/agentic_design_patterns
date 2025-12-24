# Based on original work by Marco Fago (MIT License)
# Modified by Islam Amin, 2025

import os
import uuid
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# LLM based routing

# Load environment variables BEFORE importing google modules
load_dotenv()

# Check for API key - Google ADK reads from GOOGLE_API_KEY automatically
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    print("  Add one of these to your .env file:")
    print("  GOOGLE_API_KEY=your-api-key")
    exit(1)

# Set GOOGLE_API_KEY if only GEMINI_API_KEY was provided
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event

# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents.

def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the booking was handled.
    """
    print("-------------------------- Booking Handler Called ----------------------------")
    return f"Booking action for '{request}' has been simulated."

def info_handler(request: str) -> str:
    """
    Handles general information requests.
    Args:
        request: The user's question.
    Returns:
        A message indicating the information request was handled.
    """
    print("-------------------------- Info Handler Called ----------------------------")
    return f"Information request for '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    return f"Coordinator could not delegate request: '{request}'. Please clarify."

# --- Create Tools from Functions ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# Define specialized sub-agents equipped with their respective tools
booking_agent = Agent(
    name="BookingAgent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a booking specialist. When you receive a request, "
        "use the booking_handler tool to process it. Always call the tool."
    ),
    description="Handles ONLY flight bookings and hotel reservations. Use for: 'book a flight', 'reserve a hotel', 'find flights to X'.",
    tools=[booking_tool]
)

info_agent = Agent(
    name="InfoAgent",
    model="gemini-2.0-flash",
    instruction=(
        "You are an information specialist. When you receive a question, "
        "use the info_handler tool to process it. Always call the tool."
    ),
    description="Handles ALL general questions and information requests. Use for: questions about facts, trivia, 'what is', 'tell me about', 'how does X work'.",
    tools=[info_tool]
)

# Define the parent agent with explicit delegation instructions
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "You are a routing coordinator. You MUST delegate every request to one of your sub-agents. "
        "NEVER answer questions yourself.\n\n"
        "ROUTING RULES:\n"
        "1. BookingAgent: Use ONLY for flight/hotel booking requests (e.g., 'book a flight', 'reserve hotel', 'find flights')\n"
        "2. InfoAgent: Use for EVERYTHING ELSE - all questions, facts, trivia, explanations, 'what is X', 'tell me about Y'\n\n"
        "Examples:\n"
        "- 'Book me a hotel' -> delegate to BookingAgent\n"
        "- 'What is the capital of France?' -> delegate to InfoAgent\n"
    ),
    description="A coordinator that routes user requests to the correct specialist agent.",
    sub_agents=[booking_agent, info_agent]
)

# --- Execution Logic ---

async def run_coordinator(runner: Runner, session_service: InMemorySessionService, request: str):
    """Runs the coordinator agent with a given request and delegates."""
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        
        # Properly await the async session creation
        await session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            if event.is_final_response() and event.content:
                # Try to get text directly from event.content to avoid iterating parts
                if hasattr(event.content, 'text') and event.content.text:
                     final_result = event.content.text
                elif event.content.parts:
                    # Fallback: Iterate through parts and extract text
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                break

        print(f"Coordinator Final Response: {final_result}")
        return final_result
    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


async def async_main():
    """Async main function to run the ADK example."""
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")

    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=coordinator,
        app_name="routing_example",
        session_service=session_service
    )
    
    # Example Usage
    result_a = await run_coordinator(runner, session_service, "Book me a hotel in Paris.")
    print(f"Final Output A: {result_a}")
    result_b = await run_coordinator(runner, session_service, "What is the highest mountain in the world?")
    print(f"Final Output B: {result_b}")
    result_c = await run_coordinator(runner, session_service, "Tell me a random fact.")
    print(f"Final Output C: {result_c}")
    result_d = await run_coordinator(runner, session_service, "Find flights to Tokyo next month.")
    print(f"Final Output D: {result_d}")


if __name__ == "__main__":
    asyncio.run(async_main())