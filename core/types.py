# ============================================================================
# FILE: core/types.py
# PURPOSE: Define all data structures and state management
# ============================================================================

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
import hashlib


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    Used when multiple nodes update nested structures.
    
    Example:
        left = {"compute": {"recommendation": "...old..."}}
        right = {"network": {"recommendation": "...new..."}}
        result = {"compute": {...}, "network": {...}}
    """
    result = left.copy()
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def last_value(left: Any, right: Any) -> Any:
    """Simple 'last one wins' reducer."""
    return right


class ArchitectureState(TypedDict):
    """
    The complete state for architecture generation.
    
    PHASES:
    1. GENERATION: Supervisor → Architects → Synthesizer
    2. VALIDATION: Validator Supervisor → Validators → Validation Synthesizer
    3. DECISION: Iterate or Finish?
    4. FINAL: Generate final document
    """
    
    # ========== METADATA ==========
    messages: Annotated[List, add]
    user_problem: Annotated[str, last_value]
    iteration_count: Annotated[int, last_value]
    min_iterations: Annotated[int, last_value]
    max_iterations: Annotated[int, last_value]
    
    # ========== GENERATION PHASE ==========
    architecture_domain_tasks: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    architecture_components: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    proposed_architecture: Annotated[Dict[str, Any], merge_dicts]
    
    # ========== VALIDATION PHASE ==========
    validation_feedback: Annotated[List[Dict[str, Any]], add]
    validation_summary: Annotated[Optional[str], last_value]
    
    # ========== DECISION ==========
    has_validation_errors: Annotated[bool, lambda a, b: b]
    
    # ========== FINAL OUTPUT ==========
    final_architecture: Annotated[Optional[Dict[str, Any]], last_value]
    architecture_summary: Annotated[Optional[str], last_value]


def create_initial_state(
    user_problem: str,
    min_iterations: int = 2,
    max_iterations: int = 3
) -> ArchitectureState:
    """Create the initial state for a new architecture generation run."""
    from langchain_core.messages import HumanMessage
    
    return {
        "messages": [HumanMessage(content=user_problem)],
        "user_problem": user_problem,
        "iteration_count": 0,
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
        "architecture_domain_tasks": {},
        "architecture_components": {},
        "proposed_architecture": {},
        "validation_feedback": [],
        "validation_summary": None,
        "has_validation_errors": False,
        "final_architecture": None,
        "architecture_summary": None
    }