# Code Quality Crew with Goal Setting and Monitoring Pattern
# uv add crewai litellm langfuse

import os
import sys
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Import goal tracking from local module
from .goal_tracker import (
    GoalTracker,
    create_goal_tools,
    get_default_tracker,
    reset_default_tracker,
)

# Load environment variables
load_dotenv()

# --- Session Management ---
# Sessions group related traces together in Langfuse
# Set LANGFUSE_SESSION_ID in .env to reuse the same session across runs
# Set LANGFUSE_USER_ID to track by user
DEFAULT_USER_ID = os.getenv("LANGFUSE_USER_ID", "default-user")
DEFAULT_SESSION_ID = os.getenv("LANGFUSE_SESSION_ID")  # None if not set


def get_or_generate_session_id(prefix: str = "crew") -> str:
    """Get session ID from env or generate a new one."""
    if DEFAULT_SESSION_ID:
        return DEFAULT_SESSION_ID
    # Generate a unique session ID
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{prefix}-{timestamp}-{short_uuid}"


# --- Langfuse Observability Setup ---
# Set these in your .env file:
# LANGFUSE_ENABLED=1
# LANGFUSE_PUBLIC_KEY=your_public_key
# LANGFUSE_SECRET_KEY=your_secret_key
# LANGFUSE_HOST=http://localhost:3000 (or https://cloud.langfuse.com)

# Check if Langfuse is configured
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED") == "1" and bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

langfuse_client = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import Langfuse
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        
        # Initialize Langfuse client
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=langfuse_host,
        )
        
        # Verify connection
        if langfuse_client.auth_check():
            print(f"✅ Langfuse: Connected to {langfuse_host}")
            
            # Initialize OpenInference instrumentation for hierarchical traces
            CrewAIInstrumentor().instrument(skip_dep_check=True)
            LiteLLMInstrumentor().instrument()
            print("   Hierarchical tracing enabled via OpenInference instrumentation")
        else:
            print("⚠️ Langfuse: Auth check failed - check your credentials")
            LANGFUSE_ENABLED = False
        
    except ImportError as e:
        print(f"⚠️ Missing packages: {e}")
        print("   Run: uv add langfuse openinference-instrumentation-crewai openinference-instrumentation-litellm")
        LANGFUSE_ENABLED = False
    except Exception as e:
        print(f"⚠️ Langfuse setup error: {e}")
        LANGFUSE_ENABLED = False
else:
    print("ℹ️ Langfuse: Not configured. Set LANGFUSE_ENABLED=1, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Set up LLMs (Groq and Gemini) ---
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    print("❌ Error: GROQ_API_KEY not found in .env")
    sys.exit(1)

if not gemini_api_key:
    print("❌ Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in .env")
    sys.exit(1)

# Configure Groq LLM for CrewAI (fast inference)
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
)

# Configure Gemini LLM for CrewAI (Google's model)
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_api_key,
)


# =============================================================================
# GOAL TRACKER INSTANCE & TOOLS
# =============================================================================

# Create a goal tracker for this session
goal_tracker = GoalTracker()

# Create tools bound to this tracker
set_coding_goal, update_progress, get_goal_status, quality_check = create_goal_tools(goal_tracker)


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

# 1. The Prompt Refiner Agent (uses Groq for fast, reliable inference)
prompt_refiner = Agent(
    role='Prompt Refiner & Goal Setter',
    goal='Transform user requirements into clear, actionable coding goals with specific acceptance criteria. Ensure all requirements are captured and refined for optimal AI interaction.',
    backstory="""You are an expert at understanding user intent and transforming vague 
    requirements into crystal-clear specifications. You have years of experience in 
    software requirements engineering and know how to ask the right questions to 
    eliminate ambiguity. You set the foundation for success by creating well-defined 
    goals that the entire team can rally around. You always use the Goal Setting Tool 
    to formally register goals.""",
    verbose=True,
    allow_delegation=True,
    tools=[set_coding_goal, get_goal_status],
    llm=groq_llm,  # Groq for reliable tool handling
)

# 2. The Peer Programmer Agent
peer_programmer = Agent(
    role='Senior Peer Programmer',
    goal='Write clean, efficient, and well-structured code that meets all specified goals and acceptance criteria. Collaborate and brainstorm solutions.',
    backstory="""You are a senior software engineer with 15+ years of experience across 
    multiple programming languages and paradigms. You excel at turning requirements into 
    elegant code solutions. You believe in writing code that is not just functional but 
    also readable and maintainable. You always consider edge cases and follow best 
    practices. You update goal progress as you work.""",
    verbose=True,
    allow_delegation=True,
    tools=[update_progress, quality_check],
    llm=groq_llm,
)

# 3. The Code Reviewer Agent
code_reviewer = Agent(
    role='Expert Code Reviewer',
    goal='Review code for bugs, performance issues, security vulnerabilities, and adherence to best practices. Provide constructive feedback and suggestions for improvement.',
    backstory="""You are a meticulous code reviewer with a keen eye for detail. You've 
    reviewed thousands of pull requests and have prevented countless bugs from reaching 
    production. You understand that good code review is not about finding faults but 
    about improving code quality collaboratively. You check code against the acceptance 
    criteria and update goal status based on your findings.""",
    verbose=True,
    allow_delegation=True,
    tools=[update_progress, quality_check, get_goal_status],
    llm=groq_llm,
)

# 4. The Documenter Agent
documenter = Agent(
    role='Technical Documentation Specialist',
    goal='Generate clear, comprehensive, and user-friendly documentation including docstrings, API documentation, and usage examples.',
    backstory="""You are a technical writer who believes that great code deserves great 
    documentation. You have the rare ability to explain complex technical concepts in 
    simple terms. You write documentation that serves both beginners and experienced 
    developers. You ensure every function, class, and module is properly documented 
    with examples.""",
    verbose=True,
    allow_delegation=False,
    tools=[update_progress],
    llm=groq_llm,
)

# 5. The Test Writer Agent
test_writer = Agent(
    role='Quality Assurance Engineer',
    goal='Create comprehensive unit tests, integration tests, and edge case tests that ensure code reliability and correctness.',
    backstory="""You are a QA engineer who lives by the mantra "if it's not tested, 
    it's broken." You have deep expertise in test-driven development, behavior-driven 
    development, and various testing frameworks. You write tests that not only verify 
    functionality but also serve as living documentation of how the code should behave. 
    You ensure high test coverage and meaningful assertions.""",
    verbose=True,
    allow_delegation=False,
    tools=[update_progress, quality_check],
    llm=groq_llm,
)

# 6. The Project Monitor Agent (Orchestrator - uses Groq for reliable tool handling)
project_monitor = Agent(
    role='Project Monitor & Quality Gate',
    goal='Monitor overall progress, ensure all goals are met, coordinate between agents, and provide final quality assessment.',
    backstory="""You are the team lead who keeps everyone aligned and on track. You have 
    a holistic view of the project and ensure nothing falls through the cracks. You are 
    responsible for the final sign-off on deliverables and making sure all acceptance 
    criteria are met before declaring a goal complete. You report on progress and 
    identify any blockers.""",
    verbose=True,
    allow_delegation=True,
    tools=[get_goal_status, update_progress],
    llm=groq_llm,  # Groq for reliable tool handling (Gemini had empty response issues)
)


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

def create_tasks(user_request: str, use_case: str):
    """Create tasks based on user's coding request and use case."""
    
    # Task 1: Refine the prompt and set goals
    refine_and_set_goals_task = Task(
        description=f"""
        Analyze the user's coding request and use case, then create clear, actionable goals.
        
        USER REQUEST: {user_request}
        USE CASE: {use_case}
        
        Your responsibilities:
        1. Break down the request into specific, measurable goals
        2. Use the Goal Setting Tool to formally register each goal
        3. Define clear acceptance criteria for each goal
        4. Consider edge cases and potential challenges
        5. Ensure the goals are achievable and well-scoped
        
        Create at least one goal for: Code Implementation, Testing, and Documentation.
        """,
        expected_output="""
        A comprehensive goal plan including:
        - List of registered goals with IDs
        - Acceptance criteria for each goal
        - Any clarifications or refinements made to the original request
        - Suggested approach for the team
        """,
        agent=prompt_refiner,
    )
    
    # Task 2: Write the code
    write_code_task = Task(
        description=f"""
        Based on the refined goals, write high-quality code that meets all requirements.
        
        Original Request: {user_request}
        Use Case: {use_case}
        
        Your responsibilities:
        1. Review the goals set by the Prompt Refiner using the Status Report Tool
        2. Write clean, efficient code that addresses all acceptance criteria
        3. Use meaningful variable and function names
        4. Include inline comments for complex logic
        5. Follow language-specific best practices
        6. Update progress using the Progress Update Tool as you work
        7. Run quality checks on your code
        
        IMPORTANT: Your final answer MUST include the complete code in a Python code block.
        """,
        expected_output="""
        The complete Python code implementation in a code block:
        
        ```python
        # Complete implementation with:
        # - All required functions/classes
        # - Proper error handling
        # - Docstrings and inline comments
        # - Production-ready quality
        ```
        
        Include a brief explanation of how the code works.
        """,
        agent=peer_programmer,
        context=[refine_and_set_goals_task],
    )
    
    # Task 3: Review the code
    review_code_task = Task(
        description="""
        Perform a thorough code review of the code produced by the Peer Programmer.
        
        Your responsibilities:
        1. Check the current goal status using the Status Report Tool
        2. Review code for:
           - Bugs and logical errors
           - Performance issues
           - Security vulnerabilities
           - Code style and readability
           - Adherence to acceptance criteria
        3. Run all quality checks (syntax, logic, style, security)
        4. Provide specific, actionable feedback
        5. Update goal status based on your review:
           - Mark as 'completed' if all criteria are met
           - Mark as 'needs_revision' if changes are required
        6. If revisions needed, clearly specify what changes are required
        """,
        expected_output="""
        A comprehensive code review including:
        - List of issues found (if any) with severity levels
        - Specific suggestions for improvement
        - Quality check results
        - Final verdict: APPROVED or NEEDS_REVISION
        - Updated goal statuses
        """,
        agent=code_reviewer,
        context=[write_code_task],
    )
    
    # Task 4: Write documentation
    write_documentation_task = Task(
        description="""
        Create comprehensive documentation for the code.
        
        Your responsibilities:
        1. Write clear docstrings for all functions and classes
        2. Create a module-level documentation explaining:
           - Purpose of the code
           - How to use it
           - Dependencies and requirements
        3. Include usage examples
        4. Document any configuration options
        5. Update progress using the Progress Update Tool
        
        Make the documentation accessible to both beginners and experienced developers.
        """,
        expected_output="""
        Complete documentation including:
        - Module/package overview
        - Function/class documentation with docstrings
        - Usage examples with expected output
        - Any setup or configuration instructions
        - Common pitfalls or gotchas
        """,
        agent=documenter,
        context=[write_code_task, review_code_task],
    )
    
    # Task 5: Write tests
    write_tests_task = Task(
        description="""
        Create comprehensive tests for the code.
        
        Your responsibilities:
        1. Write unit tests for all functions/methods
        2. Include edge case tests
        3. Write integration tests if applicable
        4. Ensure tests are:
           - Independent and isolated
           - Clearly named and documented
           - Covering positive and negative scenarios
        5. Run quality checks on test code
        6. Update progress using the Progress Update Tool
        
        IMPORTANT: Your final answer MUST include the complete test code in a Python code block.
        """,
        expected_output="""
        The complete test suite in a code block:
        
        ```python
        import unittest
        # or import pytest
        
        # Complete test implementation with:
        # - Unit tests for all functions
        # - Edge case tests
        # - Clear test names
        ```
        
        Brief instructions for running the tests (e.g., pytest or python -m unittest).
        """,
        agent=test_writer,
        context=[write_code_task, review_code_task],
    )
    
    # Task 6: Final monitoring and quality gate
    final_monitoring_task = Task(
        description="""
        Perform final project monitoring and quality assessment.
        
        Your responsibilities:
        1. Review the overall goal status using the Status Report Tool
        2. Ensure all goals are marked as completed
        3. Verify all deliverables are present in the context from previous tasks
        4. Create a final summary report
        5. IMPORTANT: You MUST copy and include the COMPLETE code, documentation, and tests 
           from the context into your final answer. Do not summarize - include the full code.
        
        This is the quality gate - nothing ships without your approval.
        Your final answer MUST contain the actual code, not just a summary.
        """,
        expected_output="""
        A complete final deliverable that includes:
        
        ## Summary
        - Brief summary of completed goals
        - Quality assessment: SHIP IT or HOLD
        
        ## Code (REQUIRED - include the full implementation)
        ```python
        # The complete code from the Peer Programmer
        ```
        
        ## Documentation (REQUIRED - include the full docs)
        The complete documentation from the Documenter
        
        ## Tests (REQUIRED - include the full test suite)
        ```python
        # The complete tests from the Test Writer
        ```
        
        ## Recommendations
        Any future improvements suggested
        """,
        agent=project_monitor,
        context=[write_code_task, review_code_task, write_documentation_task, write_tests_task],
    )
    
    return [
        refine_and_set_goals_task,
        write_code_task,
        review_code_task,
        write_documentation_task,
        write_tests_task,
        final_monitoring_task,
    ]


# =============================================================================
# CREW DEFINITION
# =============================================================================

def create_code_quality_crew(user_request: str, use_case: str) -> Crew:
    """
    Creates and returns the Code Quality Crew configured for the given request.
    
    Args:
        user_request: The user's coding request/requirements
        use_case: Description of the use case for the code
        
    Returns:
        Configured Crew ready to execute
    """
    tasks = create_tasks(user_request, use_case)
    
    crew = Crew(
        agents=[
            prompt_refiner,
            peer_programmer,
            code_reviewer,
            documenter,
            test_writer,
            project_monitor,
        ],
        tasks=tasks,
        process=Process.sequential,  # Tasks execute in order
        verbose=True,
        memory=True,  # Enable memory for context retention
        planning=False,  # Disabled - sequential process + task contexts already coordinate workflow
        # Use Google Generative AI embeddings for memory (avoids OpenAI embeddings requirement)
        embedder={
            "provider": "google-generativeai",
            "config": {
                "model": "models/text-embedding-004",
                "api_key": gemini_api_key,
            }
        },
    )
    
    return crew


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the Code Quality Crew."""
    
    print("\n" + "="*60)
    print("🚀 CODE QUALITY CREW - Goal Setting & Monitoring Pattern")
    print("="*60)
    
    # Example user request - you can modify this
    user_request = """
    Create a Python function that validates email addresses using regex.
    The function should:
    - Accept a string input
    - Return True if valid email, False otherwise
    - Handle edge cases like empty strings
    - Support common email formats
    """
    
    use_case = """
    This will be used in a user registration system where we need to 
    validate email addresses before storing them in the database.
    It needs to be robust and handle various email formats from 
    international users.
    """
    
    print("\n📋 USER REQUEST:")
    print("-" * 40)
    print(user_request)
    print("\n📌 USE CASE:")
    print("-" * 40)
    print(use_case)
    print("\n" + "="*60)
    
    # Create and run the crew
    crew = create_code_quality_crew(user_request, use_case)
    
    print("\n🏃 Starting Crew Execution...")
    print("-" * 60)
    
    # Execute crew with Langfuse tracing if enabled
    if LANGFUSE_ENABLED and langfuse_client:
        session_id = get_or_generate_session_id("code-quality")
        print(f"📊 Langfuse Session: {session_id}")
        
        # Create outer span with input/output - instrumentation creates nested child spans
        with langfuse_client.start_as_current_span(
            name="code-quality-crew",
            input={"request": user_request, "use_case": use_case},
        ) as span:
            # Set session and user on the trace
            span.update_trace(session_id=session_id, user_id=DEFAULT_USER_ID)
            
            result = crew.kickoff()
            
            span.update(output={"final_result": str(result), "goals": goal_tracker.get_status_report()})
        
        langfuse_client.flush()
        print(f"\n📡 Traces sent to Langfuse!")
    else:
        result = crew.kickoff()
    
    print("\n" + "="*60)
    print("✅ CREW EXECUTION COMPLETE")
    print("="*60)
    print("\n📊 FINAL RESULT:")
    print("-" * 60)
    print(result)
    
    # Print final goal status
    print("\n" + goal_tracker.get_status_report())


def interactive_mode():
    """
    Interactive mode to get user input for custom coding requests.
    """
    print("\n" + "="*60)
    print("🚀 CODE QUALITY CREW - Interactive Mode")
    print("="*60)
    
    print("\n📝 Enter your coding request (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    user_request = "\n".join(lines)
    
    print("\n📌 Enter the use case (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    use_case = "\n".join(lines)
    
    if not user_request.strip():
        print("❌ No request provided. Exiting.")
        return
    
    if not use_case.strip():
        use_case = "General purpose coding task."
    
    # Create and run the crew
    crew = create_code_quality_crew(user_request, use_case)
    
    print("\n🏃 Starting Crew Execution...")
    print("-" * 60)
    
    # Execute crew with Langfuse tracing if enabled
    if LANGFUSE_ENABLED and langfuse_client:
        session_id = get_or_generate_session_id("interactive")
        print(f"📊 Langfuse Session: {session_id}")
        
        # Create outer span with input/output - instrumentation creates nested child spans
        with langfuse_client.start_as_current_span(
            name="interactive-crew",
            input={"request": user_request, "use_case": use_case},
        ) as span:
            # Set session and user on the trace
            span.update_trace(session_id=session_id, user_id=DEFAULT_USER_ID)
            
            result = crew.kickoff()
            
            span.update(output={"final_result": str(result), "goals": goal_tracker.get_status_report()})
        
        langfuse_client.flush()
        print(f"\n📡 Traces sent to Langfuse!")
    else:
        result = crew.kickoff()
    
    print("\n" + "="*60)
    print("✅ CREW EXECUTION COMPLETE")
    print("="*60)
    print("\n📊 FINAL RESULT:")
    print("-" * 60)
    print(result)
    
    # Print final goal status
    print("\n" + goal_tracker.get_status_report())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()

