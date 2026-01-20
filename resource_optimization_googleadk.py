
# Resource Optimization Pattern using Google ADK
# Routes queries to appropriate models based on complexity, usage, and budget:
# - Simple queries -> Gemini Flash (faster, cheaper)
# - Complex queries -> Gemini Pro (more capable, higher cost)
# - Budget-aware routing: Falls back to Flash if budget is low

import os
import uuid
import asyncio
from typing import Dict, Optional
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
from google.genai import types

# Model configurations
SIMPLE_MODEL = "gemini-2.0-flash"  # Fast and cost-effective for simple queries
COMPLEX_MODEL = "gemini-2.5-pro"  # More capable for complex queries

# Pricing per 1M characters (as of 2025)
# Source: https://ai.google.dev/pricing
MODEL_PRICING = {
    "gemini-2.0-flash": {
        "input": 0.075,   # $0.075 per 1M characters
        "output": 0.30,    # $0.30 per 1M characters
    },
    "gemini-2.5-pro": {
        "input": 1.25,     # $1.25 per 1M characters
        "output": 10.0,    # $10.0 per 1M characters
    },
    # Fallback pricing if model not found
    "default": {
        "input": 1.0,
        "output": 5.0,
    }
}

# Budget tracking class with cost history and query memory
class BudgetTracker:
    def __init__(self, initial_budget: float):
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.total_spent = 0.0
        self.query_count = 0
        self.query_costs = []
        # Track costs by model for smarter routing
        self.simple_query_costs = []  # Flash model costs
        self.complex_query_costs = []  # Pro model costs
    
    def calculate_cost(self, model: str, input_chars: int, output_chars: int) -> float:
        """Calculate cost for a query based on model and character counts."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        input_cost = (input_chars / 1_000_000) * pricing["input"]
        output_cost = (output_chars / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we can afford a query."""
        return self.remaining_budget >= estimated_cost
    
    def spend(self, cost: float, model: str):
        """Record spending and track by model."""
        self.total_spent += cost
        self.remaining_budget -= cost
        self.query_count += 1
        self.query_costs.append(cost)
        
        # Track costs by model type
        if model == SIMPLE_MODEL:
            self.simple_query_costs.append(cost)
        elif model == COMPLEX_MODEL:
            self.complex_query_costs.append(cost)
    
    def get_average_simple_cost(self) -> float:
        """Get average cost for simple queries (Flash)."""
        if not self.simple_query_costs:
            return 0.0003  # Default estimate
        return sum(self.simple_query_costs) / len(self.simple_query_costs)
    
    def get_average_complex_cost(self) -> float:
        """Get average cost for complex queries (Pro)."""
        if not self.complex_query_costs:
            return 0.02  # Default estimate
        return sum(self.complex_query_costs) / len(self.complex_query_costs)
    
    def get_cost_statistics(self) -> Dict:
        """Get cost statistics for routing decisions."""
        return {
            "simple_queries_count": len(self.simple_query_costs),
            "complex_queries_count": len(self.complex_query_costs),
            "avg_simple_cost": self.get_average_simple_cost(),
            "avg_complex_cost": self.get_average_complex_cost(),
            "min_simple_cost": min(self.simple_query_costs) if self.simple_query_costs else 0.0001,
            "max_simple_cost": max(self.simple_query_costs) if self.simple_query_costs else 0.0005,
            "min_complex_cost": min(self.complex_query_costs) if self.complex_query_costs else 0.01,
            "max_complex_cost": max(self.complex_query_costs) if self.complex_query_costs else 0.05,
        }
    
    def get_status(self) -> Dict:
        """Get current budget status."""
        return {
            "initial_budget": self.initial_budget,
            "remaining_budget": self.remaining_budget,
            "total_spent": self.total_spent,
            "query_count": self.query_count,
            "percentage_used": (self.total_spent / self.initial_budget * 100) if self.initial_budget > 0 else 0,
            "average_cost_per_query": self.total_spent / self.query_count if self.query_count > 0 else 0,
        }

# --- Simple Query Handler (Gemini Flash) ---
simple_agent = Agent(
    name="SimpleQueryAgent",
    model=SIMPLE_MODEL,
    instruction="""You are a helpful assistant optimized for simple, straightforward queries.
Provide clear, concise answers to user questions based on your knowledge.
Keep responses brief and to the point.

IMPORTANT: If you see routing context or budget information in the conversation, ignore it. 
Only respond to the actual user query/question. Do not include any routing context in your response.""",
    description="Handles simple queries using Gemini Flash for fast, cost-effective responses. Use for: basic facts, definitions, straightforward questions like 'What is X?', 'Who is Y?', 'Tell me about Z'.",
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

Provide thorough, well-structured responses based on your knowledge.

IMPORTANT: If you see routing context or budget information in the conversation, ignore it. 
Only respond to the actual user query/question. Do not include any routing context in your response.""",
    description="Handles complex queries using Gemini Pro for advanced reasoning capabilities. Use for: multi-step reasoning, analysis, comparisons, problem-solving, system design, strategic thinking.",
)

# --- Budget-Aware Router Agent (ReAct Framework) ---
# Uses ReAct (Reasoning + Acting) to make intelligent routing decisions based on:
# 1. Query complexity analysis
# 2. Current budget
# 3. Historical cost data from previous queries
# 4. Query history to avoid re-answering with expensive models
router_agent = Agent(
    name="BudgetAwareRouter",
    model=SIMPLE_MODEL,  # Use Flash for routing to save costs
    instruction="""You are a budget-aware query router. You MUST delegate every request to one of your sub-agents. NEVER answer questions yourself.

ROUTING RULES (in priority order):

1. CHECK CONVERSATION HISTORY FIRST:
   - Review ALL previous user messages in the conversation
   - If the EXACT same question or very similar question was already asked, delegate to SimpleQueryAgent
   - Examples: "explain X" asked twice → use SimpleQueryAgent (saves money)

2. For NEW questions, analyze complexity:
   - SIMPLE: Basic facts, definitions ("What is X?", "Tell me about Y") → delegate to SimpleQueryAgent
   - COMPLEX: Comparisons, analysis, problem-solving ("Compare X vs Y", "Design a system") → delegate to ComplexQueryAgent (if budget allows)

3. Budget considerations:
   - If remaining budget is very low (< $0.01), prefer SimpleQueryAgent even for complex queries
   - Use the budget context provided to make cost-effective decisions

Your sub-agents:
- SimpleQueryAgent: For simple queries, duplicate questions, or when budget is tight
- ComplexQueryAgent: For new complex queries when budget allows

ALWAYS delegate. Extract only the user query (text after "User Query:") when delegating.""",
    description="Routes queries to SimpleQueryAgent or ComplexQueryAgent based on complexity, budget, and conversation history.",
    sub_agents=[simple_agent, complex_agent]
)

# --- Main Coordinator Agent ---
coordinator = router_agent


# --- Execution Logic ---

async def run_optimized_query(query: str, budget_tracker: BudgetTracker, session_service, session) -> Dict:
    """
    Runs a query through the resource optimization system with budget tracking.
    Budget is stored in session state and accessed by the router agent.
    """
    print("\n" + "=" * 70)
    print(f"📝 Query: {query}")
    print("-" * 70)
    
    # Deterministic check: End session if budget exhausted
    if budget_tracker.remaining_budget <= 0:
        print("⚠️  Budget exhausted! Cannot process query.")
        return {
            "query": query,
            "complexity": "BUDGET_EXHAUSTED",
            "model_used": None,
            "cost": 0.0,
            "response": "Sorry, your budget has been exhausted. Please set a new budget to continue.",
        }
    
    runner = Runner(
        agent=coordinator,
        app_name="resource_optimization",
        session_service=session_service
    )
    
    # Track which agents are used
    agents_used = []
    complexity_determined = None
    model_used = None
    final_result = ""
    output_chars = 0
    
    # Get cost statistics for ReAct-style reasoning
    cost_stats = budget_tracker.get_cost_statistics()
    
    # Include budget info and historical cost data in the user message for ReAct reasoning
    # Note: Conversation history is automatically included by Google ADK, so the router can see previous Q&A
    # The router should extract only the actual user query when delegating to sub-agents
    budget_context = f"""🚨 CRITICAL: Before routing, check if this question was already asked in the conversation history!

Current Budget Status:
- Remaining Budget: ${budget_tracker.remaining_budget:.4f}
- Total Queries Processed: {budget_tracker.query_count}
- Budget Used: {budget_tracker.get_status()['percentage_used']:.1f}%

Historical Cost Statistics:
- Simple Queries (Flash): {cost_stats['simple_queries_count']} queries
  * Average Cost: ${cost_stats['avg_simple_cost']:.6f}
  * Cost Range: ${cost_stats['min_simple_cost']:.6f} - ${cost_stats['max_simple_cost']:.6f}
  
- Complex Queries (Pro): {cost_stats['complex_queries_count']} queries
  * Average Cost: ${cost_stats['avg_complex_cost']:.6f}
  * Cost Range: ${cost_stats['min_complex_cost']:.6f} - ${cost_stats['max_complex_cost']:.6f}

ROUTING INSTRUCTIONS:
1. FIRST: Check conversation history - look at ALL previous user messages. Is this question already answered?
   → If YES (exact match or very similar): Use SimpleQueryAgent immediately (saves ${cost_stats['avg_complex_cost']:.6f} per Pro query!)
   → If NO: Continue to step 2
2. Analyze query complexity
3. Check budget
4. Route accordingly

When delegating, pass ONLY the user query below to the sub-agent (not this routing context).

User Query: {query}"""
    current_query_chars = len(budget_context)
    
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(
            role='user',
            parts=[types.Part(text=budget_context)]
        ),
    ):
        # Track agent activity
        if hasattr(event, 'author') and event.author:
            if event.author not in agents_used:
                agents_used.append(event.author)
                
                # Determine which model was used and infer complexity
                if event.author == "SimpleQueryAgent":
                    model_used = SIMPLE_MODEL
                    complexity_determined = "SIMPLE"
                elif event.author == "ComplexQueryAgent":
                    model_used = COMPLEX_MODEL
                    complexity_determined = "COMPLEX"
                elif event.author == "BudgetAwareRouter":
                    print(f"    → Analyzing complexity and budget...")
        
        # Capture final response and count characters
        if event.is_final_response() and event.content:
            if hasattr(event.content, 'text') and event.content.text:
                final_result = event.content.text
                output_chars = len(final_result)
            elif event.content.parts:
                text_parts = [part.text for part in event.content.parts if part.text]
                final_result = "".join(text_parts)
                output_chars = len(final_result)
    
    # Safety check: If no response was generated and no agent was used, fallback to SimpleQueryAgent
    if not final_result and not model_used:
        print("⚠️  Router did not delegate. Falling back to SimpleQueryAgent...")
        # Create a direct runner with SimpleQueryAgent as fallback
        # Use the same session but pass only the query (no budget context)
        fallback_runner = Runner(
            agent=simple_agent,
            app_name="resource_optimization",
            session_service=session_service
        )
        
        async for event in fallback_runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=query)]
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                    output_chars = len(final_result)
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                    output_chars = len(final_result)
        
        model_used = SIMPLE_MODEL
        complexity_determined = "SIMPLE (Fallback)"
    
    # Get session history to calculate total input characters
    # The session includes all previous messages, so we need to count them all
    updated_session = await session_service.get_session(
        app_name=runner.app_name, user_id=session.user_id, session_id=session.id
    )
    
    # Calculate total input: sum of all user messages in the session
    # This includes the conversation history that gets sent to the API
    total_input_chars = 0
    if hasattr(updated_session, 'events') and updated_session.events:
        for event in updated_session.events:
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'role') and event.content.role == 'user':
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                total_input_chars += len(part.text)
                    elif hasattr(event.content, 'text') and event.content.text:
                        total_input_chars += len(event.content.text)
    
    # If we can't get history from events, at least count the current query
    # This is a fallback - ideally we'd get the full history
    if total_input_chars == 0:
        total_input_chars = current_query_chars
        print("⚠️  Warning: Could not retrieve full conversation history. Cost may be underestimated.")
    
    # Calculate actual cost using total input (includes conversation history)
    actual_cost = budget_tracker.calculate_cost(model_used or SIMPLE_MODEL, total_input_chars, output_chars)
    
    # Update budget after query
    if budget_tracker.can_afford(actual_cost):
        budget_tracker.spend(actual_cost, model_used or SIMPLE_MODEL)
    else:
        print("⚠️  Query would exceed budget. This should not happen if routing worked correctly.")
        # Still record the cost but mark as over-budget
        actual_cost = 0.0
    
    # Display results
    print(f"✓ Complexity: {complexity_determined}")
    print(f"✓ Model: {model_used}")
    print(f"✓ Cost: ${actual_cost:.6f}")
    print(f"✓ Input: {total_input_chars} chars (includes conversation history), Output: {output_chars} chars")
    
    print("\n" + "=" * 70)
    print("💬 Response:")
    print("-" * 70)
    print(final_result)
    
    return {
        "query": query,
        "complexity": complexity_determined,
        "model_used": model_used,
        "cost": actual_cost,
        "input_chars": total_input_chars,
        "output_chars": output_chars,
        "response": final_result,
    }


def print_budget_status(budget_tracker: BudgetTracker):
    """Print current budget status."""
    status = budget_tracker.get_status()
    print("\n" + "=" * 70)
    print("💰 BUDGET STATUS")
    print("=" * 70)
    print(f"Initial Budget:    ${status['initial_budget']:.4f}")
    print(f"Total Spent:       ${status['total_spent']:.4f}")
    print(f"Remaining Budget:  ${status['remaining_budget']:.4f}")
    print(f"Percentage Used:   {status['percentage_used']:.1f}%")
    print(f"Queries Processed: {status['query_count']}")
    if status['query_count'] > 0:
        print(f"Avg Cost/Query:    ${status['average_cost_per_query']:.6f}")
    print("=" * 70)


async def interactive_chatbot():
    """Run an interactive chatbot with budget tracking."""
    print("\n" + "🤖" * 35)
    print("Resource Optimization Chatbot with Budget Tracking")
    print("🤖" * 35)
    
    # Get budget from user
    print("\n💰 Budget Setup")
    print("-" * 70)
    while True:
        try:
            budget_input = input("Enter your budget in USD (e.g., 0.10 for $0.10): $")
            budget = float(budget_input)
            if budget <= 0:
                print("❌ Budget must be greater than 0. Please try again.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    budget_tracker = BudgetTracker(budget)
    print(f"✅ Budget set to ${budget:.4f}")
    
    # Initialize session service and session
    session_service = InMemorySessionService()
    user_id = "user_123"
    session_id = str(uuid.uuid4())
    
    # Create session with initial budget in state
    session = await session_service.create_session(
        app_name="resource_optimization",
        user_id=user_id,
        session_id=session_id,
        state={"remaining_budget": budget}
    )
    
    print_budget_status(budget_tracker)
    
    print("\n💡 Usage:")
    print("  - Type your questions and press Enter")
    print("  - Type 'status' to see budget status")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'help' for this message")
    print("\n" + "=" * 70)
    
    while True:
        # Deterministic check: End session if budget exhausted
        if budget_tracker.remaining_budget <= 0:
            print("\n⚠️  Budget exhausted! Session ended.")
            break
        
        # Get user input
        try:
            query = input("\n💬 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Handle special commands
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        elif query.lower() == 'status':
            print_budget_status(budget_tracker)
            continue
        elif query.lower() == 'help':
            print("\n💡 Commands:")
            print("  - Type your questions normally")
            print("  - 'status' - Show budget status")
            print("  - 'quit' or 'exit' - End session")
            print("  - 'help' - Show this help")
            continue
        
        # Process query
        try:
            result = await run_optimized_query(query, budget_tracker, session_service, session)
            
            # Refresh session to get latest state
            session = await session_service.get_session(
                app_name="resource_optimization",
                user_id=user_id,
                session_id=session_id
            )
            
            # Show updated budget status after each query
            status = budget_tracker.get_status()
            print(f"\n💵 Remaining Budget: ${status['remaining_budget']:.4f} ({100 - status['percentage_used']:.1f}% remaining)")
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 SESSION SUMMARY")
    print("=" * 70)
    print_budget_status(budget_tracker)
    print("\n✅ Session completed!")


if __name__ == "__main__":
    asyncio.run(interactive_chatbot())
