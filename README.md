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
| **Chaining** | Sequential processing where output flows to the next step | `chaining.py` | - |

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

### 4. Chaining Pattern
Passes output sequentially through a pipeline of agents:
```
Agent 1 → Agent 2 → Agent 3 → Final Output
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
```

### Running Examples

```bash
# Run with uv
uv run routing_googleadk.py
uv run parallelization_langchain.py
uv run reflection_googleadk.py

# Or with python directly
python routing_googleadk.py
```

---

## 📖 Learning Notes

### Key Takeaways

1. **Google ADK vs LangChain**: Both frameworks have their strengths. Google ADK provides structured agent hierarchies (SequentialAgent, ParallelAgent, LoopAgent), while LangChain offers flexible runnable compositions.

3. **Loop Control**: In Google ADK, use the `exit_loop` tool to break out of a `LoopAgent` - simply outputting "stop" won't work.

4. **State Management**: Initialize all state variables before entering loops to avoid `KeyError` on template substitution.

---

## 🙏 Acknowledgments

- **Antonio** for writing [Agentic Design Patterns](https://www.amazon.com/Agentic-Design-Patterns-Hands-Intelligent/dp/3032014018) - the inspiration for this learning journey
- **Google** for the Agent Development Kit (ADK)
- **LangChain** team for their excellent framework

---

## 📝 License

This is a personal learning project. Feel free to use the code for your own learning!

---

*Built with curiosity and ☕ by Islam Amin*

