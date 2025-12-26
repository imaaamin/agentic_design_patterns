# Based on original work by Marco Fago (MIT License)
# Modified by Islam Amin, 2025
#
# Reflection Pattern using Google ADK's LoopAgent with exit_loop tool

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

from google.adk.agents import SequentialAgent, LlmAgent, LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import exit_loop
from google.genai import types

GEMINI_MODEL = "gemini-2.0-flash"


# --- 0. Initialize state variables (runs once before loop) ---
init_feedback = LlmAgent(
    name="InitFeedback",
    model=GEMINI_MODEL,
    instruction="""Output exactly: INITIAL""",
    output_key="critic_feedback"
)

init_draft = LlmAgent(
    name="InitDraft",
    model=GEMINI_MODEL,
    instruction="""Output exactly: No code yet.""",
    output_key="current_draft"
)

init_state = SequentialAgent(
    name="InitState",
    sub_agents=[init_feedback, init_draft]
)

# --- 1. Generator (writes or revises code) ---
generator = LlmAgent(
    name="CodeGenerator",
    model=GEMINI_MODEL,
    instruction="""You are a Python programmer.

Previous feedback: {critic_feedback}
Current code: {current_draft}

- If feedback is "INITIAL": write NEW code for the user's request.
- If feedback starts with "REVISE": improve the code based on the feedback.

Output ONLY the Python code, nothing else.""",
    output_key="current_draft"
)

# --- 2. Critic (reviews and can exit loop when satisfied) ---
critic = LlmAgent(
    name="CodeCritic",
    model=GEMINI_MODEL,
    instruction="""You are a senior Python engineer. Review this code:

CODE:
{current_draft}

Evaluate for: correctness, efficiency, readability, and best practices. Be very meticulous and detailed in your evaluation.

DO NOT JUST ACCEPT THE FIRST DRAFT. YOU MUST PROVIDE FEEDBACK AND IMPROVEMENTS.

If the code is EXCELLENT and production-ready:
- Call the exit_loop tool to stop the revision cycle
- Then say: APPROVED: [brief praise]

If the code needs improvement:
- Do NOT call exit_loop
- Respond: REVISE: [specific improvements needed]

Be demanding - only approve truly polished code.""",
    tools=[exit_loop],
    output_key="critic_feedback"
)

# --- 3. One iteration: Generate → Critique ---
one_iteration = SequentialAgent(
    name="GenerateAndCritique",
    sub_agents=[generator, critic]
)

# --- 4. LoopAgent - Repeats until exit_loop is called ---
reflection_loop = LoopAgent(
    name="ReflectionLoop",
    sub_agents=[one_iteration],
    max_iterations=5
)

# --- 5. Final Output ---
final_output = LlmAgent(
    name="FinalOutput",
    model=GEMINI_MODEL,
    instruction="""Present the final approved code nicely formatted:

FINAL CODE:
{current_draft}

APPROVAL:
{critic_feedback}"""
)

# --- 6. Complete Pipeline ---
# init_state runs first to set critic_feedback, then loop, then final output
reflection_pipeline = SequentialAgent(
    name="ReflectionPipeline",
    sub_agents=[init_state, reflection_loop, final_output]
)


# --- Execution ---

async def run_reflection(topic: str):
    print("=" * 60)
    print("Google ADK LoopAgent Reflection Demo")
    print("=" * 60)
    print(f"\nTask: {topic}")
    print("\nFlow: Generate → Critique → (repeat until exit_loop called)")
    print()
    
    session_service = InMemorySessionService()
    runner = Runner(
        agent=reflection_pipeline,
        app_name="reflection_demo",
        session_service=session_service
    )
    
    user_id = "user_123"
    session_id = str(uuid.uuid4())
    
    await session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    
    iteration = 0
    final_result = ""
    
    print("🔄 Starting reflection loop...")
    print("-" * 40)
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role='user',
            parts=[types.Part(text=topic)]
        ),
    ):
        if hasattr(event, 'author') and event.author:
            if event.author == "CodeGenerator":
                iteration += 1
                print(f"\n📝 Iteration {iteration}: Generating code...")
            elif event.author == "CodeCritic":
                print(f"🔍 Critic reviewing...")
                if event.actions and event.actions.escalate:
                    print(f"✅ Approved - exiting loop!")
            elif event.author == "FinalOutput":
                print(f"\n📋 Preparing final output...")
        
        if event.is_final_response() and event.content:
            if hasattr(event.content, 'text') and event.content.text:
                final_result = event.content.text
            elif event.content.parts:
                text_parts = [part.text for part in event.content.parts if part.text]
                final_result = "".join(text_parts)
    
    print("-" * 40)
    print(f"\n📊 Total iterations: {iteration}")
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULT:")
    print("=" * 60)
    print(final_result)


if __name__ == "__main__":
    asyncio.run(run_reflection("Write python code to calculate the factorial of a number"))
