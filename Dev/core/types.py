# ============================================================================
# FILE: core/types.py
# PURPOSE: Define all data structures and state management
# ============================================================================

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
import hashlib


# ============================================================================
# UNDERSTANDING STATE IN LANGGRAPH
# ============================================================================
# State is the "shared memory" between all nodes in your graph.
# Think of it as a whiteboard that every node can read from and write to.
#
# KEY CONCEPT: Reducers
# Each field in state has a "reducer" - a function that decides how to merge
# updates from different nodes.
#
# Examples:
# - add_messages: Appends new messages (used for conversation history)
# - last_value: Overwrites with the new value (used for final results)
# - custom reducer: Your own logic (e.g., merge dictionaries)
# ============================================================================


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    Used when multiple nodes update nested structures (e.g., architecture components).
    
    Example:
        left = {"compute": {"recommendation": "...old..."}}
        right = {"network": {"recommendation": "...new..."}}
        result = {"compute": {...}, "network": {...}}  # Both present!
    
    Right takes precedence if same key exists.
    """
    result = left.copy()
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)  # Recursive merge
        else:
            result[key] = value
    return result


def last_value(left: Any, right: Any) -> Any:
    """
    Simple 'last one wins' reducer.
    Used for fields that should only have one value at a time.
    
    Example: iteration_count - we only care about the current count
    """
    return right


def validation_feedback_reducer(left: List[Any], right: List[Any]) -> List[Any]:
    """
    Smart reducer for validation feedback.
    - If right is empty [], it's a RESET signal (from supervisor)
    - Otherwise, append items while deduplicating by domain
    
    Why? Validators run in parallel and send feedback. We want to:
    1. Collect all feedback in a list
    2. Avoid duplicates (same domain validated twice = overwrite)
    3. Allow reset between iterations
    
    Example flow:
        Iteration 1: feedback = [compute_feedback, network_feedback]
        After iteration: feedback = []  # Reset signal
        Iteration 2: feedback = [compute_feedback_improved]
    """
    # Reset detection: empty list from supervisor
    if right == [] and left != []:
        return []
    
    if not right:
        return left
    if not left:
        return right
    
    # Deduplication by domain
    feedback_dict = {}
    for item in left:
        if isinstance(item, dict):
            domain = item.get('domain', 'unknown')
            # Use hash to detect duplicate content
            result = item.get('validation_result', '')
            result_hash = hashlib.md5(result.encode()).hexdigest()[:8]
            key = f"{domain}_{result_hash}"
            feedback_dict[key] = item
    
    for item in right:
        if isinstance(item, dict):
            domain = item.get('domain', 'unknown')
            result = item.get('validation_result', '')
            result_hash = hashlib.md5(result.encode()).hexdigest()[:8]
            key = f"{domain}_{result_hash}"
            feedback_dict[key] = item  # Newer overwrites
    
    return list(feedback_dict.values())


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ArchitectureState(TypedDict):
    """
    The complete state for architecture generation.
    
    Each field represents a different aspect of the system:
    - Metadata fields: User input, iteration tracking
    - Generation fields: What the architects create
    - Validation fields: What validators find
    - Error fields: Tracking problems for iteration loop
    
    IMPORTANT: The Annotated type hint INCLUDES the reducer function.
    This tells LangGraph how to merge updates from parallel nodes.
    """
    
    # ========== METADATA ==========
    # These track the overall process
    
    messages: Annotated[List, add]
    # Reducer: add = append messages (built-in, from langgraph.graph)
    # Why: We want to preserve conversation history
    
    user_problem: Annotated[str, last_value]
    # Reducer: last_value = keep only the latest value
    # Why: User problem doesn't change during execution
    
    iteration_count: Annotated[int, last_value]
    # Reducer: last_value
    # Why: We only care about current iteration number
    
    min_iterations: Annotated[int, last_value]
    max_iterations: Annotated[int, last_value]
    # Reducer: last_value
    # Why: These config values don't change
    
    
    # ========== ARCHITECTURE GENERATION ==========
    # These are created by the architect nodes
    
    architecture_domain_tasks: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    # Reducer: merge_dicts
    # Why: Multiple task decompositions happen over iterations
    # Example: {"compute": {...}, "network": {...}, "storage": {...}}
    
    architecture_components: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    # Reducer: merge_dicts
    # Why: Each architect (compute, network, storage, database) updates independently
    # The merge_dicts reducer combines: {"compute": {...}, "network": {...}}
    #
    # HOW IT WORKS (Parallel Execution):
    # Step 1: compute_architect returns {"architecture_components": {"compute": {...}}}
    # Step 2: network_architect returns {"architecture_components": {"network": {...}}}
    # Step 3: storage_architect returns {"architecture_components": {"storage": {...}}}
    # Pregel (LangGraph runtime) merges all three using merge_dicts
    # Result: {"architecture_components": {"compute": {...}, "network": {...}, "storage": {...}}}
    
    proposed_architecture: Annotated[Dict[str, Any], merge_dicts]
    # Reducer: merge_dicts
    # Why: Synthesizer creates a complete architecture summary
    
    
    # ========== VALIDATION ==========
    # These are created by the validator nodes
    
    validation_feedback: Annotated[List[Dict[str, Any]], validation_feedback_reducer]
    # Reducer: validation_feedback_reducer (custom)
    # Why: We need to collect feedback from 4 parallel validators
    # AND detect/fix duplicates AND allow reset between iterations
    #
    # Example with 4 parallel validators:
    # compute_validator returns: {"validation_feedback": [{"domain": "compute", ...}]}
    # network_validator returns: {"validation_feedback": [{"domain": "network", ...}]}
    # storage_validator returns: {"validation_feedback": [{"domain": "storage", ...}]}
    # database_validator returns: {"validation_feedback": [{"domain": "database", ...}]}
    # Custom reducer combines: [all_four_feedback_items]
    
    validation_summary: Annotated[Optional[str], last_value]
    # Reducer: last_value
    # Why: Only one summary per validation phase
    
    audit_feedback: Annotated[List[Dict[str, Any]], add]
    # Reducer: add (built-in, append)
    # Why: Keep complete history of ALL feedback across iterations
    # This is for audit trail/debugging
    
    
    # ========== ERROR TRACKING ==========
    # These determine if we iterate or finish
    
    factual_errors_exist: Annotated[bool, lambda a, b: b]
    # Reducer: lambda a, b: b (simple overwrite)
    # Why: Validators set this. Last value wins. We reset it each iteration.
    # Example: Validator says "has_errors: True" -> field becomes True
    #          Supervisor resets to False at start of next iteration
    
    design_flaws_exist: Annotated[bool, lambda a, b: a or b]
    # Reducer: lambda a, b: a or b (logical OR)
    # Why: Once a design flaw is found, it stays True across iterations
    # (Actually not used currently, but kept for future use)
    
    
    # ========== FINAL OUTPUT ==========
    
    final_architecture: Annotated[Optional[Dict[str, Any]], last_value]
    # Reducer: last_value
    # Why: Only one final architecture document
    
    architecture_summary: Annotated[Optional[str], last_value]
    # Reducer: last_value
    # Why: Only one summary string


# ============================================================================
# KEY INSIGHT: HOW REDUCERS ENABLE PARALLEL EXECUTION
# ============================================================================
#
# Without reducers, parallel updates would conflict:
#
# WITHOUT reducers (BROKEN):
#   - compute_architect writes: state["architecture_components"] = {"compute": {...}}
#   - network_architect writes: state["architecture_components"] = {"network": {...}}
#   - Result: ONLY network component exists (compute overwritten!)
#
# WITH merge_dicts reducer (CORRECT):
#   - compute_architect writes: {"architecture_components": {"compute": {...}}}
#   - network_architect writes: {"architecture_components": {"network": {...}}}
#   - Pregel runtime calls: merge_dicts(old_state, update1)
#   - Then calls: merge_dicts(result, update2)
#   - Result: {"architecture_components": {"compute": {...}, "network": {...}}}
#
# This is how LangGraph handles parallel node execution!
#
# ============================================================================