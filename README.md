# Agentic Design Patterns

> 🎓 **Islam Amin's Learning Journey** into the world of AI Agents

This repository documents my exploration of agentic design patterns for building intelligent AI systems. The code and concepts here are mostly copied and modified code from this excellent book:

📚 [**Agentic Design Patterns: Hands-On Intelligent AI**](https://www.amazon.com/Agentic-Design-Patterns-Hands-Intelligent/dp/3032014018) by **Antonio Gulli**

---

## 🎯 What Are Agentic Design Patterns?

Agentic design patterns are reusable architectural approaches for building AI agents that can reason, plan, and take actions autonomously. These patterns help structure complex AI workflows in maintainable and scalable ways.

---

## 📁 Patterns Implemented

Each pattern is implemented using two frameworks for comparison:
- **Google ADK** (Agent Development Kit)
- **LangChain**

| Pattern | Description | Google ADK | LangChain |
|---------|-------------|------------|-----------|
| **Routing** | Dynamically routes requests to specialized agents based on intent | `routing_googleadk.py` | `routing_langchain.py` |
| **Parallelization** | Executes multiple agents concurrently for faster results | `parallelization_googleadk.py` | `parallelization_langchain.py` |
| **Reflection** | Iterative self-improvement through generate-critique loops | `reflection_googleadk.py` | `reflection_langchain.py` |
| **Resource Optimization** | Routes queries to appropriate models based on complexity (cost optimization) | `resource_optimization_googleadk.py` | - |
| **Chaining** | Sequential processing where output flows to the next step | `chaining.py` | - |
| **Peer Review (MCP)** | Document analysis with Docling MCP + dynamic tool discovery | - | `peer_review_docling_mcp.py` |
| **Peer Review (Direct)** | Document analysis with direct Docling + multimodal support | - | `peer_review_docling_langgraph.py` |

---

## 🔄 Pattern Overviews

### 1. Routing Pattern
Routes user requests to the appropriate specialized agent:
```
User Request → Router Agent → [Booking Agent | Info Agent | ...]
```

### 2. Parallelization Pattern  
Runs multiple agents simultaneously and combines their results:
```
                ┌→ Agent 1 →┐
User Request →  ├→ Agent 2 →├→ Combine Results
                └→ Agent 3 →┘
```

### 3. Reflection Pattern
Iteratively improves output through self-critique:
```
Generator → Critic → (if not approved) → Generator → Critic → ... → Final Output
```

### 4. Resource Optimization Pattern
Routes queries to appropriate models based on complexity to optimize costs:
```
User Query → Complexity Analyzer → [Simple Query Agent (Flash) | Complex Query Agent (Pro)]
```

**Key Benefits:**
- **Cost Optimization**: Uses faster, cheaper models (Gemini Flash) for simple queries
- **Quality Assurance**: Uses more capable models (Gemini Pro) only when needed for complex reasoning
- **Automatic Routing**: Analyzes query complexity and routes automatically

**Use Cases:**
- High-volume query systems where cost matters
- Applications with mixed query complexity
- Systems needing to balance cost and quality

### 5. Chaining Pattern
Passes output sequentially through a pipeline of agents:
```
Agent 1 → Agent 2 → Agent 3 → Final Output
```

### 6. Peer Review Pattern (Reflection + MCP)
Combines the Reflection pattern with document processing for automated peer review of research papers:

```
PDF → Docling Parser → Producer Agent → Critic Agent → (iterate) → Final Review
```

This pattern demonstrates:
- **Docling** for intelligent PDF parsing (extracts text, tables, figures)
- **MCP (Model Context Protocol)** for dynamic tool discovery
- **langchain-mcp-adapters** for LangChain ↔ MCP integration
- **Multimodal support** for analyzing figures/images in papers

---

## 📄 Peer Review Workflow (Docling + MCP)

A real-world application of the Reflection pattern for automated peer review of research papers.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Peer Review Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PDF ──→ Docling MCP Server ──→ Markdown Content            │
│              (via MCP tools)                                 │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────────────────────────────┐                │
│  │         LangGraph Workflow              │                │
│  │  ┌──────────┐      ┌──────────┐        │                │
│  │  │ Producer │ ←──→ │  Critic  │        │                │
│  │  │  Agent   │      │  Agent   │        │                │
│  │  └──────────┘      └──────────┘        │                │
│  │       ↑                  │              │                │
│  │       └──────────────────┘              │                │
│  │         (iterate until approved)        │                │
│  └─────────────────────────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│              Final Peer Review                               │
└─────────────────────────────────────────────────────────────┘
```

### Two Implementations

#### 1. `peer_review_docling_mcp.py` - MCP Mode (Dynamic Tool Discovery)

Uses **langchain-mcp-adapters** to dynamically discover and use Docling MCP tools. The agent decides which tools to call at runtime.

**Requires HTTP transport** for persistent connections (stdio spawns new processes per tool call, losing document cache):

```bash
# Terminal 1: Start the MCP server
uvx --from docling-mcp docling-mcp-server --transport http --port 8000

# Terminal 2: Run peer review
uv run peer_review_docling_mcp.py paper.pdf
```

**Key Features:**
- Dynamic tool discovery via MCP protocol
- Agent autonomously decides tool usage
- Persistent HTTP connection maintains document cache

#### 2. `peer_review_docling_langgraph.py` - Direct Mode

Uses Docling directly without MCP. More reliable and supports multimodal analysis.

```bash
uv run peer_review_docling_langgraph.py paper.pdf --multimodal
```

**Key Features:**
- Direct Docling API (no MCP overhead)
- **Multimodal support**: Extracts and analyzes figures/images
- Base64 encodes images for Gemini's vision capabilities

### Technical Learnings

#### MCP Transport Issues

**Problem:** stdio transport spawns new server processes per tool call, causing document cache loss.

```
Tool 1 (convert) → Server Instance A (creates document key)
Tool 2 (export)  → Server Instance B (document key not found!)
```

**Solution:** Use HTTP transport with persistent server:
```python
mcp_config = {
    "docling": {
        "url": "http://localhost:8000/mcp/",
        "transport": "streamable_http",
    }
}
```

#### Environment Variable Conflicts

**Problem:** docling-mcp uses pydantic-settings which reads `.env` and rejects unknown variables.

**Solution:** Run server from directory without `.env`:
```python
server_params = StdioServerParameters(
    command="uvx",
    args=["--from", "docling-mcp", "docling-mcp-server"],
    cwd=os.path.expanduser("~")  # No .env here
)
```

#### langchain-mcp-adapters API Changes

```python
# Old API (deprecated):
async with MultiServerMCPClient(config) as client:
    tools = client.get_tools()

# New API (0.2.x):
client = MultiServerMCPClient(config)
tools = await client.get_tools()
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repo
git clone https://github.com/imaaamin/agentic_design_patterns.git
cd agentic_design_patterns

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with your API keys:

```env
GOOGLE_API_KEY=your-google-api-key
GROQ_API_KEY=your-groq-api-key

# For Vertex AI Search (optional)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

### Additional Dependencies for Peer Review

```bash
# Docling for document parsing
uv add docling

# LangGraph for workflow orchestration  
uv add langgraph langchain-google-genai

# MCP adapters for dynamic tool discovery
uv add langchain-mcp-adapters mcp
```

### Running Examples

```bash
# Basic patterns
uv run routing_googleadk.py
uv run parallelization_langchain.py
uv run reflection_googleadk.py
uv run resource_optimization_googleadk.py

# Peer Review (Direct mode - recommended)
uv run peer_review_docling_langgraph.py paper.pdf
uv run peer_review_docling_langgraph.py paper.pdf --multimodal  # With image analysis

# Peer Review (MCP mode - requires server)
# Terminal 1: uvx --from docling-mcp docling-mcp-server --transport http --port 8000
# Terminal 2:
uv run peer_review_docling_mcp.py paper.pdf
```

---

## 📖 Learning Notes

### Key Takeaways

1. **Google ADK vs LangChain**: Both frameworks have their strengths. Google ADK provides structured agent hierarchies (SequentialAgent, ParallelAgent, LoopAgent), while LangChain offers flexible runnable compositions.

2. **Loop Control**: In Google ADK, use the `exit_loop` tool to break out of a `LoopAgent` - simply outputting "stop" won't work.

3. **State Management**: Initialize all state variables before entering loops to avoid `KeyError` on template substitution.

4. **MCP Transport Matters**: When using MCP with stateful servers (like docling-mcp), use HTTP transport instead of stdio. Stdio spawns new processes per tool call, losing in-memory state like document caches.

5. **Direct vs MCP**: For document processing, direct library usage (e.g., `from docling import DocumentConverter`) is often more reliable than MCP. Use MCP when you need dynamic tool discovery or multi-server orchestration.

6. **Multimodal Reviews**: When reviewing documents with figures, extract images and send them alongside text to vision-capable models (like Gemini) for comprehensive analysis.

---

## 🙏 Acknowledgments

- **Mohammad Amin** for mentoring me through this journey
- **Antonio** for writing [Agentic Design Patterns](https://www.amazon.com/Agentic-Design-Patterns-Hands-Intelligent/dp/3032014018)
- **Google** for the Agent Development Kit (ADK) and Gemini
- **LangChain** team for their excellent framework and MCP adapters
- **IBM/Docling** team for the document parsing library
- **Anthropic** for the Model Context Protocol (MCP) specification

---

## 📝 License

This is a personal learning project. Feel free to use the code for your own learning!

---

*Built with curiosity and ☕ by Islam Amin*

