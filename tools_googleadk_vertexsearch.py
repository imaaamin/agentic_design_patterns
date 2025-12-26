import asyncio
import warnings
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import VertexAiSearchTool
import os
from dotenv import load_dotenv

# Suppress Google Cloud SDK credential warnings
warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")

# --- Configuration ---
# VertexAiSearchTool requires Vertex AI (Google Cloud), NOT the Gemini API.
# You must authenticate using Application Default Credentials (ADC):
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Run: gcloud auth application-default login
#   3. Set your project: gcloud config set project YOUR_PROJECT_ID
#
# Required environment variables:
#   GOOGLE_CLOUD_PROJECT: Your Google Cloud project ID
#   GOOGLE_CLOUD_LOCATION: Region (e.g., "us-central1" or "global")
#   DATASTORE_ID: Full datastore path in format:
#     "projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}"

load_dotenv()

# Vertex AI requires project and location, NOT an API key
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
DATASTORE_ID = os.environ.get("DATASTORE_ID")

if not GOOGLE_CLOUD_PROJECT:
    print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable is not set.")
    print("   Set it to your Google Cloud project ID (e.g., 'gen-lang-client-0907105907')")
    exit(1)

# Tell the genai library to use Vertex AI instead of Gemini API
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
# Set quota project to avoid warnings
os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = GOOGLE_CLOUD_PROJECT

# --- Application Constants ---
APP_NAME = "vsearch_app"
USER_ID = "user_123"  # Example User ID
SESSION_ID = "session_456" # Example Session ID

# --- Tool and Agent Definition ---
# Create the Vertex AI Search tool with the datastore ID
print(f"📦 Configuring VertexAiSearchTool with datastore:")
print(f"   {DATASTORE_ID}")
vsearch_tool = VertexAiSearchTool(data_store_id=DATASTORE_ID)

# Create an Agent that uses the Vertex AI Search tool
# Using Vertex AI model format
vsearch_agent = Agent(
    name="avalanche_coin_docs_agent",
    description="Answers questions about Avalanche Coin Docs.",
    model="gemini-2.0-flash",
    instruction="You MUST use the Vertex AI Search tool to search the datastore for every question. Always search first, then answer based on the search results.",
    tools=[vsearch_tool]
)
print(f"✅ Agent '{vsearch_agent.name}' created with {len(vsearch_agent.tools)} tool(s)")

# --- Session Service (shared across calls) ---
session_service = InMemorySessionService()

# --- Runner Initialization ---
runner = Runner(
    agent=vsearch_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# --- Agent Invocation Logic ---
async def call_vsearch_agent_async(query: str, session_id: str, debug: bool = False):
    """Sends a query to the agent using an existing session."""
    print(f"\n{'='*50}")
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)

    try:
        # Construct the message content correctly
        content = types.Content(role='user', parts=[types.Part(text=query)])
        response_text = ""

        # Process events as they arrive from the asynchronous runner
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content
        ):
            # For token-by-token streaming of the response text
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(part.text, end="", flush=True)
                        response_text += part.text

            # Process the final response and its associated metadata
            if event.is_final_response():
                print() # Newline after the streaming response
                
                if hasattr(event, 'grounding_metadata') and event.grounding_metadata:
                    gm = event.grounding_metadata
                    retrieval_queries = getattr(gm, 'retrieval_queries', None)
                    chunks = getattr(gm, 'grounding_chunks', None)
                    
                    if retrieval_queries:
                        print(f"\n  ✅ DATASTORE WAS QUERIED!")
                        print(f"  📝 Search queries sent: {retrieval_queries}")
                        
                        if chunks:
                            print(f"  ✅ GOT {len(chunks)} RESULTS FROM DATASTORE")
                            for i, chunk in enumerate(chunks[:5]):  # Show first 5
                                print(f"\n     --- Result [{i+1}] ---")
                                # Check for retrieved_context (Vertex AI Search datastore)
                                if hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                                    ctx = chunk.retrieved_context
                                    print(f"     🗄️  SOURCE: VERTEX AI SEARCH DATASTORE")
                                    if hasattr(ctx, 'uri') and ctx.uri:
                                        print(f"     📎 URI: {ctx.uri}")
                                    if hasattr(ctx, 'title') and ctx.title:
                                        print(f"     📄 Title: {ctx.title}")
                                    if hasattr(ctx, 'document_name') and ctx.document_name:
                                        print(f"     📁 Document: {ctx.document_name}")
                                    if hasattr(ctx, 'text') and ctx.text:
                                        # Show snippet of retrieved text
                                        snippet = ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text
                                        print(f"     📝 Text: {snippet}")
                                # Check for web (Google Search)
                                elif hasattr(chunk, 'web') and chunk.web:
                                    web = chunk.web
                                    print(f"     🌐 SOURCE: WEB SEARCH (not datastore)")
                                    if hasattr(web, 'uri') and web.uri:
                                        print(f"     🔗 URL: {web.uri}")
                                    if hasattr(web, 'title') and web.title:
                                        print(f"     📄 Title: {web.title}")
                                else:
                                    print(f"     ❓ Unknown chunk type: {chunk}")
                        else:
                            print(f"  ❌ DATASTORE RETURNED 0 RESULTS")
                            print(f"     → Datastore might be empty, deleted, or query didn't match any documents")
                            print(f"     → Model is answering from its own knowledge (NOT from your data)")
                    else:
                        print(f"\n  ⚠️  No retrieval queries - datastore was NOT queried")
                else:
                    print("  ⚠️  No grounding metadata found - tool may not have been called")
                
                if not response_text:
                    print("  ⚠️  No text response received")
                    
                print("-" * 50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 50)

# --- Run Example ---
async def run_vsearch_example():
    # Create the session first (required for InMemorySessionService)
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Replace with questions relevant to YOUR datastore content
    await call_vsearch_agent_async("What is the Avalanche Coin?", session.id)
    await call_vsearch_agent_async("What are the different chains that avalanche supports and how do they work?", session.id)
    await call_vsearch_agent_async("What safety procedures are mentioned for lab X?", session.id)

# --- Execution ---
if __name__ == "__main__":
    if not DATASTORE_ID:
        print("Error: DATASTORE_ID environment variable is not set.")
    else:
        try:
            asyncio.run(run_vsearch_example())
        except RuntimeError as e:
            # This handles cases where asyncio.run is called in an environment
            # that already has a running event loop (like a Jupyter notebook).
            if "cannot be called from a running event loop" in str(e):
                print("Skipping execution in a running event loop. Please run this script directly.")
            else:
                raise e