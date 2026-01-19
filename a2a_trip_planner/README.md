# A2A Trip Planner Example

**A2A (Agent-to-Agent) Protocol Implementation**

This example demonstrates the official [A2A Protocol](https://github.com/a2aproject/A2A) with three AI agents built using different frameworks that communicate via Agent Cards and JSON-RPC 2.0.

## 🌐 What is A2A?

The [Agent2Agent (A2A) Protocol](https://a2a-protocol.org/) is an open protocol enabling communication and interoperability between opaque agentic applications. Key features:

- **Agent Cards**: JSON metadata describing agent capabilities and endpoints
- **Discovery**: Agents find each other via `.well-known/agent.json`
- **JSON-RPC 2.0**: Standardized communication over HTTP(S)
- **Opaque Collaboration**: Agents work together without exposing internals

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                          │
│              "Plan a trip from NYC to Paris"                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR (LangGraph) - Port 8001               │
│                                                                 │
│   1. Receives client request via JSON-RPC 2.0                   │
│   2. Discovers specialist agents via Agent Cards                │
│   3. Delegates tasks to specialists                             │
│   4. Synthesizes final recommendation                           │
│                                                                 │
│   Agent Card: http://localhost:8001/.well-known/agent.json      │
└───────────┬─────────────────────────────────────┬───────────────┘
            │                                     │
            │ GET /.well-known/agent.json         │ GET /.well-known/agent.json
            │ POST / (message/send)               │ POST / (message/send)
            ▼                                     ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│   FLIGHT AGENT (CrewAI)   │   │   HOTEL AGENT (Google ADK)    │
│   Port 8003               │   │   Port 8002                   │
│                           │   │                               │
│   Skills: find_flights    │   │   Skills: find_hotels         │
│                           │   │                               │
│   /.well-known/agent.json │   │   /.well-known/agent.json     │
└───────────────────────────┘   └───────────────────────────────┘
```

## 📁 Project Structure

```
a2a_trip_planner/
├── __init__.py
├── __main__.py
├── run.py                           # Client & health check utility
├── a2a_common.py                    # Shared A2A utilities
├── README.md
└── agents/
    ├── langgraph_orchestrator/
    │   ├── agent_card.json          # A2A Agent Card
    │   └── orchestrator.py          # LangGraph-based coordinator
    ├── googleadk_hotels/
    │   ├── agent_card.json          # A2A Agent Card
    │   └── hotel_agent.py           # Google ADK hotel search
    └── crewai_flights/
        ├── agent_card.json          # A2A Agent Card
        └── flight_agent.py          # CrewAI flight search
```

## 🚀 Quick Start

### Prerequisites

1. Install dependencies:
```bash
cd agentic_patterns
uv sync
uv add a2a-sdk httpx uvicorn
```

2. Set environment variables:
```bash
# Required for LLMs
export GOOGLE_API_KEY=your-google-api-key
# or
export GEMINI_API_KEY=your-gemini-api-key

# Optional for CrewAI (can use Gemini instead)
export GROQ_API_KEY=your-groq-api-key
```

### Start the Agents

**Important**: Start agents in this order (specialist agents first, then orchestrator):

**Terminal 1 - Hotel Agent (Google ADK):**
```bash
cd agentic_patterns
uv run python -m a2a_trip_planner.agents.googleadk_hotels.hotel_agent
```

**Terminal 2 - Flight Agent (CrewAI):**
```bash
cd agentic_patterns
uv run python -m a2a_trip_planner.agents.crewai_flights.flight_agent
```

**Terminal 3 - Orchestrator (LangGraph):**
```bash
cd agentic_patterns
uv run python -m a2a_trip_planner.agents.langgraph_orchestrator.orchestrator
```

### Test with the Client

**Terminal 4:**
```bash
cd agentic_patterns
uv run python a2a_trip_planner/run.py --client
```

## 🔧 A2A Protocol Details

### Agent Cards

Each agent exposes an Agent Card at `/.well-known/agent.json`:

```json
{
  "name": "CrewAI Flight Search Agent",
  "description": "Finds the best flight options for trips",
  "url": "http://localhost:8003",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "find_flights",
      "name": "Flight Search",
      "description": "Search for flights between cities",
      "tags": ["travel", "flights"],
      "examples": ["Find flights from NYC to Paris"]
    }
  ]
}
```

### Discovery

The orchestrator discovers specialist agents by:

1. Fetching Agent Cards from known URLs
2. Inspecting skills to understand capabilities
3. Creating clients for communication

```python
# Discovery via Agent Card
card_url = "http://localhost:8002/.well-known/agent.json"
response = await httpx.get(card_url)
agent_card = response.json()
```

### Communication (JSON-RPC 2.0)

Agents communicate via JSON-RPC 2.0:

```python
# Sending a task to an agent
payload = {
    "jsonrpc": "2.0",
    "id": "unique-id",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "Find hotels in Paris"}]
        }
    }
}
response = await httpx.post(agent_url, json=payload)
```

### Task Lifecycle

Tasks follow the A2A lifecycle:
- `submitted` → Task received
- `working` → Processing in progress
- `completed` → Successfully finished
- `failed` → Error occurred
- `canceled` → Canceled by request

## 🛠️ Implementation Details

### Flight Agent (CrewAI)

Based on the pattern from [a2a-samples](https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent_executor.py):

```python
class FlightAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.working, ...)
        
        result = self.crew_agent.search(query)
        
        await updater.add_artifact([Part(root=TextPart(text=result))])
        await updater.complete()
```

### Hotel Agent (Google ADK)

Uses Google ADK internally with A2A SDK wrapper:

```python
class HotelAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        result = await self.adk_agent.search(query)
        await updater.add_artifact([Part(root=TextPart(text=result))])
        await updater.complete()
```

### Orchestrator (LangGraph)

Coordinates the workflow:

```python
class OrchestratorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # 1. Discover agents
        await self.discovery.discover_all(SPECIALIST_AGENTS)
        
        # 2. Delegate to flight agent
        flights = await self.discovery.send_task("flights", query)
        
        # 3. Delegate to hotel agent
        hotels = await self.discovery.send_task("hotels", query)
        
        # 4. Synthesize recommendation
        recommendation = await self.synthesize(flights, hotels)
```

## 📚 References

- [A2A Protocol Specification](https://github.com/a2aproject/A2A)
- [A2A Python SDK](https://a2a-protocol.org/latest/sdk/python/)
- [A2A Samples Repository](https://github.com/a2aproject/a2a-samples)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)

## 🏆 Key Takeaways

1. **Agent Cards for Discovery**: Each agent publishes its capabilities at a well-known URL
2. **JSON-RPC 2.0**: Standard protocol for inter-agent communication
3. **Framework Agnostic**: Mix LangGraph, CrewAI, and Google ADK seamlessly
4. **Opaque Collaboration**: Agents work together without exposing internals
5. **Scalable**: Easy to add more specialist agents to the system

## 🔮 Future Improvements

- Add streaming support for real-time updates
- Implement push notifications for long-running tasks
- Add authentication schemes to Agent Cards
- Integrate with a central agent registry for dynamic discovery
