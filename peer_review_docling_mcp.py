"""
Peer Review Workflow using Docling MCP + LangGraph + langchain-mcp-adapters

This implements the Reflection Pattern with:
- Producer Agent: Writes/improves the peer review (has access to MCP tools)
- Critic Agent: Provides constructive feedback on the review

Uses langchain-mcp-adapters to dynamically discover and use Docling MCP tools.
The agent decides which tools to call - no hardcoded tool calls!

IMPORTANT: Uses HTTP transport for persistent MCP server connection.
The document cache persists across tool calls (unlike stdio which spawns new processes).

Install dependencies:
    uv add langgraph langchain-google-genai langchain-mcp-adapters

Usage:
    # Step 1: Start the docling-mcp server (in a separate terminal)
    uvx --from docling-mcp docling-mcp-server --transport http --port 8000
    
    # Step 2: Run the peer review workflow
    uv run peer_review_docling_mcp.py path/to/paper.pdf
    uv run peer_review_docling_mcp.py path/to/paper.pdf --direct  # Skip MCP, use direct Docling
    uv run peer_review_docling_mcp.py path/to/paper.pdf --port 9000  # Custom port
"""

import asyncio
import os
import sys
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# --- API Key Setup ---
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    sys.exit(1)
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key

# --- Configuration ---
MAX_ITERATIONS = 3


# --- State Definition ---
class PeerReviewState(TypedDict):
    """State for the peer review workflow."""
    paper_path: str              # Path to the PDF file
    paper_title: str             # Paper title
    current_review: str          # Current version of the peer review
    critic_feedback: str         # Feedback from the critic
    iteration: int               # Current iteration number
    is_approved: bool            # Whether the review is approved
    messages: Annotated[list, add_messages]  # Message history


# --- System Prompts ---
PRODUCER_SYSTEM_PROMPT = """You are an expert academic peer reviewer with access to document processing tools.

Your task is to write a comprehensive peer review for a research paper. You have access to tools that can:
- Convert documents to structured formats
- Export documents to markdown for reading
- Search and navigate document content

WORKFLOW:
1. First, use the 'convert_document_into_docling_document' tool with the paper path to parse it
2. Then use 'export_docling_document_to_markdown' with the document_key to get readable content
3. Read and analyze the paper content
4. Write your peer review

Your peer review MUST include:
1. **Summary**: Brief overview of the paper's main contribution
2. **Strengths**: What the paper does well (be specific with examples)
3. **Weaknesses**: Areas for improvement (be constructive)
4. **Questions**: Specific questions for the authors
5. **Minor Comments**: Typos, formatting, clarifications needed
6. **Overall Assessment**: Accept/Weak Accept/Weak Reject/Reject with justification

Be thorough, fair, and constructive. Reference specific sections when possible."""


CRITIC_SYSTEM_PROMPT = """You are a senior meta-reviewer evaluating peer reviews for quality.

Evaluate the peer review based on:
1. **Completeness**: Does it cover all required sections?
2. **Specificity**: Does it reference specific parts of the paper?
3. **Constructiveness**: Is feedback actionable and helpful?
4. **Fairness**: Is the assessment balanced and well-justified?
5. **Clarity**: Is the review well-written and easy to understand?

If the review is high quality, respond with:
"APPROVED: [brief explanation of why it's good]"

If it needs improvement, respond with:
"NEEDS IMPROVEMENT: [specific suggestions for what to fix]"

Be constructive but maintain high standards."""


# --- LLM Setup ---
def get_llm():
    """Get the LLM for agents."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
    )


# --- Direct Docling Parser (Fallback) ---
def parse_pdf_direct(pdf_path: str) -> tuple[str, str]:
    """Parse PDF using direct Docling API (fallback when MCP unavailable)."""
    print(f"📄 Parsing PDF directly: {pdf_path}")
    
    try:
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        title = result.document.name or os.path.basename(pdf_path)
        content = result.document.export_to_markdown()
        
        print(f"✅ Parsed: {title}")
        print(f"   Content length: {len(content)} characters")
        
        return title, content
    except ImportError:
        print("⚠️ Docling not installed. Using placeholder content.")
        return os.path.basename(pdf_path), f"[Content from {pdf_path}]"


# --- Agent Nodes ---
async def producer_node_with_tools(state: PeerReviewState, mcp_tools: list) -> dict:
    """
    Producer agent that uses MCP tools to read and review the paper.
    The agent dynamically decides which tools to call.
    """
    print(f"\n{'='*50}")
    print(f"📝 Producer Agent (Iteration {state['iteration'] + 1})")
    
    llm = get_llm()
    
    # Create a ReAct agent with MCP tools
    agent = create_react_agent(llm, mcp_tools)
    
    # Build the prompt
    if state["iteration"] == 0:
        # First iteration: Read the paper and write initial review
        prompt = f"""Please review the research paper at: {state['paper_path']}

IMPORTANT: First use the document tools to:
1. Convert the PDF using 'convert_document_into_docling_document' with source="{state['paper_path']}"
2. Export to markdown using 'export_docling_document_to_markdown' with the returned document_key
3. Read the content and write your comprehensive peer review

Write a thorough peer review following the format in your instructions."""
    else:
        # Subsequent iterations: Improve based on feedback
        prompt = f"""Please improve your peer review based on the critic's feedback.

PREVIOUS REVIEW:
{state['current_review']}

CRITIC'S FEEDBACK:
{state['critic_feedback']}

Revise your review to address all the feedback while maintaining a professional tone."""
    
    # Run the agent
    result = await agent.ainvoke({
        "messages": [
            SystemMessage(content=PRODUCER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
    })
    
    # Extract the final response
    final_message = result["messages"][-1]
    review_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    print(f"   Review length: {len(review_text)} characters")
    
    return {
        "current_review": review_text,
        "iteration": state["iteration"] + 1,
        "messages": [AIMessage(content=f"[Producer Iteration {state['iteration'] + 1}] {review_text[:500]}...")]
    }


async def producer_node_direct(state: PeerReviewState, paper_content: str) -> dict:
    """Producer agent using pre-parsed content (no MCP tools)."""
    print(f"\n{'='*50}")
    print(f"📝 Producer Agent (Iteration {state['iteration'] + 1})")
    
    llm = get_llm()
    
    if state["iteration"] == 0:
        prompt = f"""Please write a comprehensive peer review for the following research paper.

PAPER TITLE: {state['paper_title']}

PAPER CONTENT:
{paper_content[:50000]}  # Limit content size

Write a thorough peer review following the format in your instructions."""
    else:
        prompt = f"""Please improve your peer review based on the critic's feedback.

PREVIOUS REVIEW:
{state['current_review']}

CRITIC'S FEEDBACK:
{state['critic_feedback']}

Revise your review to address all the feedback."""
    
    messages = [
        SystemMessage(content=PRODUCER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = await llm.ainvoke(messages)
    review_text = response.content
    
    print(f"   Review length: {len(review_text)} characters")
    
    return {
        "current_review": review_text,
        "iteration": state["iteration"] + 1,
        "messages": [AIMessage(content=f"[Producer Iteration {state['iteration'] + 1}]")]
    }


async def critic_node(state: PeerReviewState) -> dict:
    """Critic agent that evaluates the peer review."""
    print(f"\n{'='*50}")
    print(f"🔍 Critic Agent (Evaluating Iteration {state['iteration']})")
    
    llm = get_llm()
    
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"""Please evaluate this peer review:

{state['current_review']}

Is this review ready for submission, or does it need improvement?""")
    ]
    
    response = await llm.ainvoke(messages)
    feedback = response.content
    
    is_approved = "APPROVED" in feedback.upper() and "NEEDS IMPROVEMENT" not in feedback.upper()
    
    print(f"   Feedback length: {len(feedback)} characters")
    print(f"   Status: {'✅ APPROVED' if is_approved else '🔄 Needs Improvement'}")
    
    return {
        "critic_feedback": feedback,
        "is_approved": is_approved,
        "messages": [AIMessage(content=f"[Critic] {'APPROVED' if is_approved else 'NEEDS IMPROVEMENT'}")]
    }


def should_continue(state: PeerReviewState) -> Literal["producer", "end"]:
    """Decide whether to continue the reflection loop."""
    if state["is_approved"]:
        print(f"\n✅ Review approved by critic!")
        return "end"
    
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n⚠️ Max iterations ({MAX_ITERATIONS}) reached")
        return "end"
    
    print(f"\n🔄 Continuing to iteration {state['iteration'] + 1}...")
    return "producer"


# --- Main Workflow ---
async def run_peer_review_with_mcp(pdf_path: str, mcp_port: int = 8000):
    """Run the peer review workflow using MCP tools via HTTP transport.
    
    IMPORTANT: Requires docling-mcp server running separately:
        uvx --from docling-mcp docling-mcp-server --transport http --port 8000
    
    HTTP transport maintains a persistent connection, so the document cache
    persists across tool calls (unlike stdio which spawns new processes).
    """
    print("🚀 Starting Peer Review Workflow (MCP HTTP Mode)")
    print("=" * 60)
    
    abs_pdf_path = os.path.abspath(pdf_path)
    
    # Configure MCP client for HTTP transport (persistent connection)
    # docling-mcp uses StreamableHTTP, endpoint is /mcp/
    mcp_url = f"http://localhost:{mcp_port}/mcp/"
    
    mcp_config = {
        "docling": {
            "url": mcp_url,
            "transport": "streamable_http",  # StreamableHTTP for docling-mcp
        }
    }
    
    print(f"🔌 Connecting to Docling MCP server at http://localhost:{mcp_port}...")
    print("   ℹ️  Make sure server is running:")
    print(f"      uvx --from docling-mcp docling-mcp-server --transport http --port {mcp_port}")
    
    try:
        # Create client and get tools (HTTP keeps persistent connection)
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()
    except Exception as e:
        print(f"\n❌ Failed to connect to MCP server: {e}")
        print(f"\n💡 Start the server first in another terminal:")
        print(f"   uvx --from docling-mcp docling-mcp-server --transport http --port {mcp_port}")
        raise
    
    tool_names = [t.name for t in tools]
    print(f"✅ Connected! Available tools ({len(tools)}): {tool_names[:5]}...")
    
    # Build the graph with MCP-aware producer
    async def producer_with_mcp(state):
        return await producer_node_with_tools(state, tools)
    
    graph = StateGraph(PeerReviewState)
    graph.add_node("producer", producer_with_mcp)
    graph.add_node("critic", critic_node)
    
    graph.set_entry_point("producer")
    graph.add_edge("producer", "critic")
    graph.add_conditional_edges("critic", should_continue, {
        "producer": "producer",
        "end": END
    })
    
    workflow = graph.compile()
    
    # Initial state
    initial_state = {
        "paper_path": abs_pdf_path,
        "paper_title": os.path.basename(pdf_path),
        "current_review": "",
        "critic_feedback": "",
        "iteration": 0,
        "is_approved": False,
        "messages": []
    }
    
    print("\n🔄 Starting reflection loop...")
    
    # Run the workflow
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state


async def run_peer_review_direct(pdf_path: str):
    """Run the peer review workflow using direct Docling parsing."""
    print("🚀 Starting Peer Review Workflow (Direct Mode)")
    print("=" * 60)
    
    # Parse the PDF first
    title, content = parse_pdf_direct(pdf_path)
    
    # Build a simpler graph without MCP tools
    async def producer_direct(state):
        return await producer_node_direct(state, content)
    
    graph = StateGraph(PeerReviewState)
    graph.add_node("producer", producer_direct)
    graph.add_node("critic", critic_node)
    
    graph.set_entry_point("producer")
    graph.add_edge("producer", "critic")
    graph.add_conditional_edges("critic", should_continue, {
        "producer": "producer",
        "end": END
    })
    
    workflow = graph.compile()
    
    # Initial state
    initial_state = {
        "paper_path": os.path.abspath(pdf_path),
        "paper_title": title,
        "current_review": "",
        "critic_feedback": "",
        "iteration": 0,
        "is_approved": False,
        "messages": []
    }
    
    print("\n🔄 Starting reflection loop...")
    
    # Run the workflow
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state


def print_final_review(state: dict):
    """Print the final peer review."""
    print("\n" + "=" * 60)
    print("📋 FINAL PEER REVIEW")
    print("=" * 60)
    print(f"\nPaper: {state['paper_title']}")
    print(f"Iterations: {state['iteration']}")
    print(f"Approved: {'Yes' if state['is_approved'] else 'No (max iterations reached)'}")
    print("-" * 60)
    print(state["current_review"])
    print("-" * 60)


# --- Entry Point ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run peer_review_docling_mcp.py <pdf_path> [options]")
        print("\nOptions:")
        print("  --direct       Skip MCP, use direct Docling parsing")
        print("  --port PORT    MCP server port (default: 8000)")
        print("\nMCP Mode (default):")
        print("  First start the server in another terminal:")
        print("    uvx --from docling-mcp docling-mcp-server --transport http --port 8000")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    use_direct = "--direct" in sys.argv
    
    # Parse --port argument
    mcp_port = 8000
    if "--port" in sys.argv:
        port_idx = sys.argv.index("--port")
        if port_idx + 1 < len(sys.argv):
            try:
                mcp_port = int(sys.argv[port_idx + 1])
            except ValueError:
                print("❌ Error: --port requires a number")
                sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    try:
        if use_direct:
            final_state = asyncio.run(run_peer_review_direct(pdf_path))
        else:
            final_state = asyncio.run(run_peer_review_with_mcp(pdf_path, mcp_port=mcp_port))
        
        print_final_review(final_state)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if not use_direct:
            print("\n💡 Tip: Try running with --direct flag to skip MCP")
