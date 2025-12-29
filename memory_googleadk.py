import asyncio
import time
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event, EventActions
import os
from dotenv import load_dotenv

"""
Google ADK Session State Management using EventActions.state_delta

The PROPER way to update session state is:
1. Using `output_key` parameter on LlmAgent (for agent's final responses)
2. Using `EventActions.state_delta` with `session_service.append_event()`

Direct modification of session.state is discouraged as it:
- Bypasses standard event processing
- Won't be recorded in session's event history
- May not be persisted properly
- Could cause concurrency issues
"""

load_dotenv()

# Setup API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    exit(1)
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key

# --- Session and Runner Setup ---
APP_NAME = "state_delta_demo"
USER_ID = "user1"
SESSION_ID = "session1"

session_service = InMemorySessionService()

# --- Create Agent ---
agent = Agent(
    name="stateful_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant. Keep responses brief.",
)

runner = Runner(
    agent=agent,
    app_name=APP_NAME,
    session_service=session_service,
)

async def update_state_with_delta(session, state_changes: dict, author: str = "system"):
    """
    The PROPER way to update state using EventActions.state_delta.
    Creates an Event with state_delta and appends it via session_service.
    """
    event = Event(
        author=author,
        actions=EventActions(state_delta=state_changes)
    )
    # This properly persists the state changes!
    await session_service.append_event(session=session, event=event)
    print(f"   📝 state_delta applied: {state_changes}")

async def chat(message: str, session):
    """Send a message and display the response."""
    print(f"\n{'='*50}")
    print(f"User: {message}")
    
    # Update state BEFORE agent runs using state_delta
    current_count = session.state.get("interaction_count", 0)
    print(f"\n🔵 [Before Agent] Interaction #{current_count + 1}")
    print(f"   Current state: {session.state}")
    
    await update_state_with_delta(session, {
        "interaction_count": current_count + 1,
        "last_interaction_start": time.time(),
        "status": "processing"
    })
    
    content = types.Content(role='user', parts=[types.Part(text=message)])
    start_time = time.time()
    
    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=content
        ):
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"Agent: {part.text}", end="", flush=True)
            
            if event.is_final_response():
                print()  # newline after response
                
    except Exception as e:
        print(f"\n❌ Error during chat: {e}")
        import traceback
        traceback.print_exc()
    
    # Update state AFTER agent runs using state_delta
    duration = time.time() - start_time
    print(f"\n🟢 [After Agent] Completed in {duration:.2f}s")
    
    # Refresh session to get latest state
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session.id)
    
    await update_state_with_delta(session, {
        "status": "idle",
        "last_interaction_duration": duration,
        "total_interactions": session.state.get("interaction_count", 0)
    })
    
    return session

async def main():
    # Create session with initial state
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"interaction_count": 0, "status": "idle"}
    )
    print(f"✅ Session created with initial state: {session.state}")
    
    # Have a conversation - state_delta updates happen properly
    session = await chat("Hello! What's 2+2?", session)
    
    # Check state after first interaction
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    print(f"\n📊 State after 1st interaction: {session.state}")
    
    session = await chat("Thanks! Now what's 10*10?", session)
    
    # Check final state
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    print(f"\n📊 Final state after 2 interactions: {session.state}")

if __name__ == "__main__":
    asyncio.run(main())