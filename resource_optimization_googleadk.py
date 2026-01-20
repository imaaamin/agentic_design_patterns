# Resource Optimization Pattern using Google ADK
# Routes queries to appropriate models based on complexity:
# - Simple queries -> Gemini Flash (faster, cheaper)
# - Complex queries -> Gemini Pro (more capable, higher cost)

import os
import uuid
import asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    exit(1)

if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# Model configurations
SIMPLE_MODEL = "gemini-2.0-flash"  # Fast and cost-effective for simple queries
COMPLEX_MODEL = "gemini-2.5-pro"  # More capable for complex queries

# --- Simple Query Handler (Gemini Flash) ---
simple_agent = Agent(
    name="SimpleQueryAgent",
    model=SIMPLE_MODEL,
    instruction="""You are a helpful assistant optimized for simple, straightforward queries.
Provide clear, concise answers to user questions based on your knowledge.
Keep responses brief and to the point.""",
    description="Handles simple queries using Gemini Flash for fast, cost-effective responses. Use for: basic facts, definitions, straightforward questions like 'What is X?', 'Who is Y?', 'Tell me about Z'.",
    # Note: Tools removed to avoid function calling conflicts with delegation
    # tools=[google_search]
)

# --- Complex Query Handler (Gemini Pro) ---
complex_agent = Agent(
    name="ComplexQueryAgent",
    model=COMPLEX_MODEL,
    instruction="""You are an advanced AI assistant optimized for complex, multi-step reasoning tasks.
You excel at:
- Analyzing and synthesizing information from multiple sources
- Providing detailed explanations and comparisons
- Problem-solving and strategic thinking
- Creative and analytical work
- Breaking down complex topics into understandable components

Provide thorough, well-structured responses based on your knowledge.""",
    description="Handles complex queries using Gemini Pro for advanced reasoning capabilities. Use for: multi-step reasoning, analysis, comparisons, problem-solving, system design, strategic thinking.",
    # Note: Tools removed to avoid function calling conflicts with delegation
    # tools=[google_search]
)

# --- Router Agent (Analyzes and Routes) ---
# This agent analyzes query complexity and routes to the appropriate handler
# It combines analysis and routing in one step to avoid state variable access issues
router_agent = Agent(
    name="QueryRouter",
    model=SIMPLE_MODEL,  # Use Flash for routing to save costs
    instruction="""You are a query router that analyzes query complexity and routes to the appropriate agent.

ANALYSIS RULES:
A query is SIMPLE if it:
- Asks for basic facts or definitions (e.g., "What is X?", "Who is Y?")
- Requires straightforward information retrieval
- Can be answered with a brief, factual response
- Doesn't require multi-step reasoning or analysis
- Examples: "What is the capital of France?", "Tell me about Python", "What's the weather today?"

A query is COMPLEX if it:
- Requires multi-step reasoning or analysis
- Needs synthesis of multiple concepts or sources
- Involves problem-solving or strategic thinking
- Requires detailed explanations or comparisons
- Needs creative or analytical work
- Examples: "Compare the pros and cons of microservices vs monoliths", 
  "Design a system architecture for a social media platform",
  "Explain how quantum computing could impact cryptography and what are the implications"

ROUTING RULES:
- For SIMPLE queries: delegate to SimpleQueryAgent
- For COMPLEX queries: delegate to ComplexQueryAgent

You MUST delegate to one of the sub-agents. Never answer the query yourself.""",
    description="Analyzes query complexity and routes to appropriate agents for optimal resource usage.",
    sub_agents=[simple_agent, complex_agent]
)

# --- Main Coordinator Agent ---
# The router agent handles both analysis and routing
coordinator = router_agent


# --- Execution Logic ---

async def run_optimized_query(query: str):
    """
    Runs a query through the resource optimization system.
    """
    print("=" * 70)
    print("Resource Optimization System - Model Selection Based on Complexity")
    print("=" * 70)
    print(f"\n📝 Query: {query}")
    print("\n🔄 Processing...")
    print("-" * 70)
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=coordinator,
        app_name="resource_optimization",
        session_service=session_service
    )
    
    user_id = "user_123"
    session_id = str(uuid.uuid4())
    
    await session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    
    # Track which agents are used
    agents_used = []
    complexity_determined = None
    model_used = None
    final_result = ""
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role='user',
            parts=[types.Part(text=query)]
        ),
    ):
        # Track agent activity
        if hasattr(event, 'author') and event.author:
            if event.author not in agents_used:
                agents_used.append(event.author)
                print(f"  ✓ {event.author} activated")
                
                # Determine which model was used and infer complexity
                if event.author == "SimpleQueryAgent":
                    model_used = SIMPLE_MODEL
                    complexity_determined = "SIMPLE"
                    print(f"    → Using {SIMPLE_MODEL} (Simple Query Handler)")
                    print(f"    → Complexity: SIMPLE")
                elif event.author == "ComplexQueryAgent":
                    model_used = COMPLEX_MODEL
                    complexity_determined = "COMPLEX"
                    print(f"    → Using {COMPLEX_MODEL} (Complex Query Handler)")
                    print(f"    → Complexity: COMPLEX")
                elif event.author == "QueryRouter":
                    print(f"    → Analyzing query complexity...")
        
        # Capture final response
        if event.is_final_response() and event.content:
            if hasattr(event.content, 'text') and event.content.text:
                final_result = event.content.text
            elif event.content.parts:
                text_parts = [part.text for part in event.content.parts if part.text]
                final_result = "".join(text_parts)
    
    # Complexity is determined by which agent was used
    # (already set in the event tracking above)
    
    print("-" * 70)
    print(f"\n📊 System Summary:")
    print(f"   Agents Used: {', '.join(agents_used)}")
    if complexity_determined:
        print(f"   Complexity: {complexity_determined}")
    if model_used:
        print(f"   Model Selected: {model_used}")
    
    print("\n" + "=" * 70)
    print("📋 RESPONSE:")
    print("=" * 70)
    print(final_result)
    print()
    
    return {
        "query": query,
        "complexity": complexity_determined,
        "model_used": model_used,
        "agents_used": agents_used,
        "response": final_result
    }


async def async_main():
    """Run multiple example queries to demonstrate resource optimization."""
    print("\n" + "🔬" * 35)
    print("Resource Optimization Demo")
    print("🔬" * 35 + "\n")
    
    # Example queries of varying complexity
    test_queries = [
        # Simple queries (should use Flash)
        "What is the capital of France?",
        "Tell me about Python programming language.",
        "What's the weather like today?",
        
        # Complex queries (should use Pro)
        "Compare the architectural patterns of microservices vs monolithic systems, including their trade-offs, scalability implications, and when to use each approach.",
        "Design a comprehensive system architecture for a real-time social media platform that needs to handle 100 million users, including data storage, caching strategies, and load balancing approaches.",
        "Explain how quantum computing could impact modern cryptography, including the implications for current encryption methods and potential solutions.",
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'='*70}\n")
        
        result = await run_optimized_query(query)
        results.append(result)
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 SUMMARY OF ALL QUERIES")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Query: {result['query'][:60]}...")
        print(f"   Complexity: {result['complexity'] or 'N/A'}")
        print(f"   Model: {result['model_used'] or 'N/A'}")
    
    print("\n" + "=" * 70)
    print("✅ Resource optimization demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(async_main())
