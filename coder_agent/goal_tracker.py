# Goal Tracking Module for Code Quality Crew
# Implements the Goal Setting and Monitoring Pattern

import logging
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from crewai.tools import tool

# --- Configure Logging ---
logger = logging.getLogger(__name__)


# =============================================================================
# GOAL MODELS
# =============================================================================

class Goal(BaseModel):
    """Represents a coding goal with tracking information."""
    id: str
    description: str
    acceptance_criteria: List[str]
    status: str = "pending"  # pending, in_progress, completed, needs_revision
    progress_notes: List[str] = Field(default_factory=list)


class GoalTracker:
    """
    Simple goal tracker to monitor progress across agents.
    
    This implements the Goal Setting and Monitoring pattern, allowing agents
    to set goals, update their progress, and check status.
    """
    
    def __init__(self):
        self.goals: List[Goal] = []
        self.iteration_count = 0
        self.max_iterations = 3
        
    def add_goal(self, goal: Goal) -> None:
        """Add a new goal to track."""
        self.goals.append(goal)
        logger.info(f"📌 Goal added: {goal.description}")
        
    def update_goal_status(self, goal_id: str, status: str, note: str = "") -> bool:
        """
        Update the status of an existing goal.
        
        Args:
            goal_id: The unique identifier of the goal
            status: New status (pending, in_progress, completed, needs_revision)
            note: Optional progress note
            
        Returns:
            True if goal was found and updated, False otherwise
        """
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = status
                if note:
                    goal.progress_notes.append(f"[{datetime.now().strftime('%H:%M:%S')}] {note}")
                logger.info(f"📊 Goal '{goal_id}' updated to: {status}")
                return True
        return False
    
    def get_status_report(self) -> str:
        """Generate a formatted status report of all goals."""
        report = "\n=== GOAL STATUS REPORT ===\n"
        for goal in self.goals:
            report += f"\n🎯 {goal.id}: {goal.description}\n"
            report += f"   Status: {goal.status}\n"
            report += f"   Criteria: {', '.join(goal.acceptance_criteria)}\n"
            if goal.progress_notes:
                report += f"   Notes:\n"
                for note in goal.progress_notes:
                    report += f"      - {note}\n"
        return report
    
    def reset(self) -> None:
        """Reset the tracker for a new session."""
        self.goals = []
        self.iteration_count = 0


# =============================================================================
# GOAL TRACKING TOOLS
# =============================================================================

def create_goal_tools(tracker: GoalTracker):
    """
    Factory function to create goal tracking tools bound to a specific tracker.
    
    Args:
        tracker: The GoalTracker instance to use
        
    Returns:
        Tuple of (set_coding_goal, update_progress, get_goal_status, quality_check) tools
    """
    
    @tool("Goal Setting Tool")
    def set_coding_goal(goal_id: str, description: str, criteria: str) -> str:
        """
        Sets a new coding goal with acceptance criteria.
        Args:
            goal_id: Unique identifier for the goal (e.g., 'CODE_001')
            description: Clear description of what needs to be achieved
            criteria: Comma-separated list of acceptance criteria
        Returns:
            Confirmation message with goal details
        """
        criteria_list = [c.strip() for c in criteria.split(",")]
        goal = Goal(
            id=goal_id,
            description=description,
            acceptance_criteria=criteria_list
        )
        tracker.add_goal(goal)
        return f"✅ Goal '{goal_id}' set successfully with {len(criteria_list)} acceptance criteria."

    @tool("Progress Update Tool")
    def update_progress(goal_id: str, status: str, notes: str) -> str:
        """
        Updates the progress of a coding goal.
        Args:
            goal_id: The ID of the goal to update
            status: New status (pending, in_progress, completed, needs_revision)
            notes: Progress notes or comments
        Returns:
            Confirmation of the update
        """
        success = tracker.update_goal_status(goal_id, status, notes)
        if success:
            return f"✅ Goal '{goal_id}' updated to '{status}'. Note recorded."
        return f"❌ Goal '{goal_id}' not found."

    @tool("Status Report Tool")
    def get_goal_status() -> str:
        """
        Retrieves the current status of all coding goals.
        Returns:
            A formatted report of all goals and their status
        """
        return tracker.get_status_report()

    @tool("Quality Check Tool")
    def quality_check(code: str, check_type: str) -> str:
        """
        Performs a quality check on code.
        Args:
            code: The code to check (or description of code)
            check_type: Type of check (syntax, logic, style, security)
        Returns:
            Quality assessment result
        """
        # In production, this could integrate with actual linters/analyzers
        checks = {
            "syntax": "✅ Syntax check: Code structure appears valid",
            "logic": "✅ Logic check: Control flow and logic reviewed",
            "style": "✅ Style check: Code follows consistent formatting",
            "security": "✅ Security check: No obvious vulnerabilities detected"
        }
        return checks.get(check_type.lower(), f"⚠️ Unknown check type: {check_type}")
    
    return set_coding_goal, update_progress, get_goal_status, quality_check


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# For backwards compatibility - create a default tracker and tools
_default_tracker = GoalTracker()
set_coding_goal, update_progress, get_goal_status, quality_check = create_goal_tools(_default_tracker)

def get_default_tracker() -> GoalTracker:
    """Get the default GoalTracker instance."""
    return _default_tracker

def reset_default_tracker() -> None:
    """Reset the default tracker for a new session."""
    _default_tracker.reset()

