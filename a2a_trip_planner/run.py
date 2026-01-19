#!/usr/bin/env python3
"""
A2A Trip Planner - Multi-Agent System Runner

This script helps you run the A2A Trip Planner system which consists of three agents:
1. Hotel Agent (Google ADK) - Port 8002
2. Flight Agent (CrewAI) - Port 8003
3. Orchestrator (LangGraph) - Port 8001

Usage:
    # Start all agents (each in separate terminal):
    uv run python -m agentic_patterns.a2a_trip_planner.agents.googleadk_hotels.hotel_agent
    uv run python -m agentic_patterns.a2a_trip_planner.agents.crewai_flights.flight_agent
    uv run python -m agentic_patterns.a2a_trip_planner.agents.langgraph_orchestrator.orchestrator

    # Or use this script to test with a client:
    uv run python agentic_patterns/a2a_trip_planner/run.py --client

Based on A2A Protocol: https://github.com/a2aproject/A2A
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()


# Agent URLs
ORCHESTRATOR_URL = "http://localhost:8001"
HOTEL_AGENT_URL = "http://localhost:8002"
FLIGHT_AGENT_URL = "http://localhost:8003"


def print_banner():
    """Print the application banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║       🌍  A2A TRIP PLANNER - Agent-to-Agent Protocol  🌍        ║
    ║                                                                  ║
    ║    Based on: https://github.com/a2aproject/A2A                  ║
    ║                                                                  ║
    ║    Agents:                                                       ║
    ║    • Orchestrator (LangGraph) - http://localhost:8001           ║
    ║    • Hotel Agent (GoogleADK)  - http://localhost:8002           ║
    ║    • Flight Agent (CrewAI)    - http://localhost:8003           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


async def discover_agent(url: str) -> dict:
    """Discover an agent by fetching its Agent Card."""
    card_url = f"{url}/.well-known/agent.json"
    print(f"🔍 Discovering agent at {card_url}...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(card_url, timeout=5)
            response.raise_for_status()
            card = response.json()
            print(f"   ✅ Found: {card.get('name', 'Unknown')}")
            print(f"   Skills: {[s.get('id') for s in card.get('skills', [])]}")
            return card
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return None


async def check_agents_health():
    """Check if all agents are running."""
    print("\n📊 Checking agent health...\n")
    
    agents = {
        "Orchestrator (LangGraph)": ORCHESTRATOR_URL,
        "Hotel Agent (GoogleADK)": HOTEL_AGENT_URL,
        "Flight Agent (CrewAI)": FLIGHT_AGENT_URL,
    }
    
    all_healthy = True
    for name, url in agents.items():
        card = await discover_agent(url)
        if not card:
            all_healthy = False
            print(f"   ⚠️ {name} is not running. Start it first!")
    
    return all_healthy


async def send_trip_request(query: str) -> dict:
    """Send a trip planning request to the orchestrator."""
    import uuid
    
    print(f"\n📤 Sending request to orchestrator...")
    print(f"   Query: {query[:100]}...")
    
    message_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [{"kind": "text", "text": query}]
            },
            "configuration": {
                "acceptedOutputModes": ["text"]
            }
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            ORCHESTRATOR_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=180  # 3 minutes for full workflow
        )
        response.raise_for_status()
        return response.json()


def extract_recommendation(result: dict) -> str:
    """Extract the recommendation text from the result."""
    # Try artifacts first
    task_result = result.get("result", {})
    artifacts = task_result.get("artifacts", [])
    
    for artifact in artifacts:
        parts = artifact.get("parts", [])
        for part in parts:
            if "text" in part:
                return part["text"]
    
    # Try status message
    status = task_result.get("status", {})
    message = status.get("message", {})
    parts = message.get("parts", [])
    for part in parts:
        if "text" in part:
            return part["text"]
    
    return json.dumps(result, indent=2)


async def run_client():
    """Run the A2A client to test the system."""
    print_banner()
    
    # Check if agents are running
    if not await check_agents_health():
        print("\n❌ Not all agents are running. Please start them first:")
        print("\nIn separate terminals, run:")
        print("  Terminal 1: uv run python -m agentic_patterns.a2a_trip_planner.agents.googleadk_hotels.hotel_agent")
        print("  Terminal 2: uv run python -m agentic_patterns.a2a_trip_planner.agents.crewai_flights.flight_agent")
        print("  Terminal 3: uv run python -m agentic_patterns.a2a_trip_planner.agents.langgraph_orchestrator.orchestrator")
        return
    
    print("\n" + "=" * 60)
    print("All agents are running! Ready to plan your trip.")
    print("=" * 60)
    
    # Get trip request
    print("\nEnter your trip request (or press Enter for demo):")
    user_input = input("> ").strip()
    
    if not user_input:
        user_input = "Plan a trip from New York to Paris from June 15-22, 2026. Budget $800 for flights and $200/night for hotels."
        print(f"\nUsing demo request: {user_input}")
    
    # Send request
    try:
        result = await send_trip_request(user_input)
        
        print("\n" + "🌟" * 30)
        print("\n📋 TRIP RECOMMENDATION")
        print("\n" + "🌟" * 30)
        
        recommendation = extract_recommendation(result)
        print(recommendation)
        
        print("\n" + "=" * 60)
        print("✅ Trip planning complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def show_architecture():
    """Display the A2A architecture."""
    print("""
    A2A Trip Planner Architecture
    =============================

    The system uses the A2A (Agent-to-Agent) protocol for inter-agent communication.
    Each agent exposes an Agent Card at /.well-known/agent.json for discovery.

    ┌─────────────────────────────────────────────────────────────────┐
    │                         CLIENT                                   │
    │              (sends trip request via JSON-RPC 2.0)              │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │              ORCHESTRATOR (LangGraph) - Port 8001               │
    │                                                                 │
    │   1. Receives client request                                    │
    │   2. Discovers specialist agents via Agent Cards                │
    │   3. Delegates tasks using JSON-RPC 2.0                         │
    │   4. Synthesizes final recommendation                           │
    │                                                                 │
    │   Agent Card: http://localhost:8001/.well-known/agent.json      │
    └───────────────┬─────────────────────────────┬───────────────────┘
                    │                             │
         ┌──────────┴──────────┐       ┌──────────┴──────────┐
         │                     │       │                     │
         ▼                     ▼       ▼                     ▼
    ┌───────────────────┐  ┌───────────────────────────────────────┐
    │   FLIGHT AGENT    │  │            HOTEL AGENT                │
    │   (CrewAI)        │  │            (GoogleADK)                │
    │   Port 8003       │  │            Port 8002                  │
    │                   │  │                                       │
    │   Skills:         │  │   Skills:                             │
    │   - find_flights  │  │   - find_hotels                       │
    │                   │  │                                       │
    │   Agent Card:     │  │   Agent Card:                         │
    │   /.well-known/   │  │   /.well-known/agent.json             │
    │   agent.json      │  │                                       │
    └───────────────────┘  └───────────────────────────────────────┘

    A2A Protocol Flow:
    ==================
    1. Client → Orchestrator: message/send (trip request)
    2. Orchestrator → Flight Agent: GET /.well-known/agent.json (discovery)
    3. Orchestrator → Hotel Agent: GET /.well-known/agent.json (discovery)
    4. Orchestrator → Flight Agent: message/send (flight search)
    5. Orchestrator → Hotel Agent: message/send (hotel search)
    6. Orchestrator → Client: Task result with recommendation

    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="A2A Trip Planner - Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check agent health and run client
  uv run python agentic_patterns/a2a_trip_planner/run.py --client

  # Show architecture
  uv run python agentic_patterns/a2a_trip_planner/run.py --architecture

  # Start individual agents (in separate terminals):
  uv run python -m agentic_patterns.a2a_trip_planner.agents.googleadk_hotels.hotel_agent
  uv run python -m agentic_patterns.a2a_trip_planner.agents.crewai_flights.flight_agent
  uv run python -m agentic_patterns.a2a_trip_planner.agents.langgraph_orchestrator.orchestrator
        """
    )
    
    parser.add_argument("--client", action="store_true", help="Run as A2A client to test the system")
    parser.add_argument("--architecture", action="store_true", help="Show the A2A architecture")
    parser.add_argument("--health", action="store_true", help="Check agent health only")
    
    args = parser.parse_args()
    
    if args.architecture:
        show_architecture()
    elif args.health:
        asyncio.run(check_agents_health())
    elif args.client:
        asyncio.run(run_client())
    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
