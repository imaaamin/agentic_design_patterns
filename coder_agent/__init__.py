# Coder Agent Package
# A CrewAI-based code quality crew implementing the Goal Setting and Monitoring pattern

"""
Coder Agent - AI-Powered Code Quality Crew

This package provides a multi-agent system for producing high-quality code
using the Goal Setting and Monitoring pattern.

Agents:
    - Prompt Refiner: Transforms requirements into clear goals
    - Peer Programmer: Writes clean, efficient code
    - Code Reviewer: Reviews for bugs and best practices
    - Documenter: Generates comprehensive documentation
    - Test Writer: Creates comprehensive test suites
    - Project Monitor: Ensures all goals are met

Usage:
    # Run with default example
    uv run -m coder_agent.code_quality_crew
    
    # Run in interactive mode
    uv run -m coder_agent.code_quality_crew --interactive
    
    # Or import and use programmatically
    from coder_agent import create_code_quality_crew
    
    crew = create_code_quality_crew(
        user_request="Create a function that...",
        use_case="This will be used for..."
    )
    result = crew.kickoff()

Environment Variables:
    Required:
        - GROQ_API_KEY: API key for Groq LLM
        - GOOGLE_API_KEY or GEMINI_API_KEY: API key for Google embeddings
    
    Optional (for Langfuse observability):
        - LANGFUSE_ENABLED=1
        - LANGFUSE_PUBLIC_KEY
        - LANGFUSE_SECRET_KEY
        - LANGFUSE_HOST (default: http://localhost:3000)
        - LANGFUSE_SESSION_ID (optional, auto-generated if not set)
        - LANGFUSE_USER_ID (default: default-user)
"""

from .goal_tracker import (
    Goal,
    GoalTracker,
    create_goal_tools,
    get_default_tracker,
    reset_default_tracker,
    # Default tools for backwards compatibility
    set_coding_goal,
    update_progress,
    get_goal_status,
    quality_check,
)

from .code_quality_crew import (
    create_code_quality_crew,
    create_tasks,
    goal_tracker,
    # Agents
    prompt_refiner,
    peer_programmer,
    code_reviewer,
    documenter,
    test_writer,
    project_monitor,
    # LLMs
    groq_llm,
    gemini_llm,
    # Entry points
    main,
    interactive_mode,
)

__version__ = "0.1.0"
__all__ = [
    # Goal tracking
    "Goal",
    "GoalTracker", 
    "create_goal_tools",
    "get_default_tracker",
    "reset_default_tracker",
    "set_coding_goal",
    "update_progress",
    "get_goal_status",
    "quality_check",
    # Crew
    "create_code_quality_crew",
    "create_tasks",
    "goal_tracker",
    # Agents
    "prompt_refiner",
    "peer_programmer",
    "code_reviewer",
    "documenter",
    "test_writer",
    "project_monitor",
    # LLMs
    "groq_llm",
    "gemini_llm",
    # Entry points
    "main",
    "interactive_mode",
]

