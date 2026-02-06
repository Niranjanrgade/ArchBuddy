# ============================================================================
# FILE: core/schemas.py
# PURPOSE: Define Pydantic models for LLM structured output
# ============================================================================

from pydantic import BaseModel, Field
from typing import List


# ============================================================================
# WHY PYDANTIC MODELS?
# ============================================================================
# LLMs can be unpredictable. Pydantic ensures the LLM output follows a schema.
# Without this, the LLM might return random text instead of structured data.
#
# Example without schema (BAD):
#   LLM: "Here are the tasks: compute is important, network is too..."
#   Your code: task.domain -> ERROR (string has no attribute 'domain')
#
# Example with schema (GOOD):
#   LLM: Forced to return {"tasks": [...], "goals": [...]}
#   Your code: task_decomposition.tasks[0].domain -> "compute" ✓
# ============================================================================


class DomainTask(BaseModel):
    """
    A single task for one domain architect.
    LLM MUST return this structure.
    """
    domain: str = Field(description="Domain name: compute, network, storage, or database")
    task_description: str = Field(description="What should this domain architect do?")
    requirements: List[str] = Field(description="Key requirements for this domain")
    deliverables: List[str] = Field(description="What should the architect produce?")


class TaskDecomposition(BaseModel):
    """
    Complete task decomposition from supervisor.
    LLM breaks down user's problem into domain-specific tasks.
    """
    user_problem: str
    decomposed_tasks: List[DomainTask]
    overall_architecture_goals: List[str]
    constraints: List[str]


class ValidationTask(BaseModel):
    """A validation task for one domain validator."""
    domain: str
    components_to_validate: List[str]
    validation_focus: str


class ValidationDecomposition(BaseModel):
    """
    Validation tasks from validator supervisor.
    What should each validator check?
    """
    validation_tasks: List[ValidationTask]