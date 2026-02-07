# ============================================================================
# FILE: core/schemas.py
# PURPOSE: Define Pydantic models for LLM structured output
# ============================================================================

from pydantic import BaseModel, Field
from typing import List


class DomainTask(BaseModel):
    """A single task for one domain architect."""
    domain: str = Field(description="Domain name: compute, network, storage, or database")
    task_description: str = Field(description="What should this domain architect do?")
    requirements: List[str] = Field(description="Key requirements for this domain")
    deliverables: List[str] = Field(description="What should the architect produce?")


class TaskDecomposition(BaseModel):
    """Complete task decomposition from supervisor."""
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
    """Validation tasks from validator supervisor."""
    validation_tasks: List[ValidationTask]