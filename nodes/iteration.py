# ============================================================================
# FILE: nodes/iteration.py
# PURPOSE: Decide whether to iterate or finish, generate final architecture
# ============================================================================

from typing import Literal
from langchain_core.messages import SystemMessage
from core.types import ArchitectureState
import logging
import time

logger = logging.getLogger(__name__)


def iteration_condition(state: ArchitectureState) -> Literal["iterate", "finish"]:
    """
    Decide: Should we iterate (refine) or finish?
    
    RULES:
    1. Haven't met minimum iterations? → iterate
    2. Have errors AND haven't hit max? → iterate
    3. Hit max iterations? → finish
    4. No errors AND met minimum? → finish
    """
    
    iteration = state.get("iteration_count", 0)
    min_iters = state.get("min_iterations", 1)
    max_iters = state.get("max_iterations", 3)
    has_errors = state.get("has_validation_errors", False)
    
    logger.info(f"Iteration Decision: {iteration}/{max_iters} (min: {min_iters}, errors: {has_errors})")
    
    # Rule 1: Minimum not reached
    if iteration < min_iters:
        logger.info("→ ITERATE (minimum not reached)")
        return "iterate"
    
    # Rule 2: Errors exist and haven't maxed out
    if has_errors and iteration < max_iters:
        logger.info("→ ITERATE (errors found, trying again)")
        return "iterate"
    
    # Rule 3: Reached maximum
    if iteration >= max_iters:
        logger.info("→ FINISH (maximum iterations reached)")
        return "finish"
    
    # Rule 4: No errors, minimum met
    logger.info("→ FINISH (validation passed)")
    return "finish"


def final_architecture_generator(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Generate final production-ready document.
    
    ROLE: Create delivery-ready architecture document.
    INPUT: All validated components and architecture
    OUTPUT: Comprehensive final document
    """
    
    logger.info("--- Final Architecture Generator ---")
    start_time = time.time()
    
    try:
        components = state.get("architecture_components", {})
        proposed = state.get("proposed_architecture", {})
        validation = state.get("validation_summary", "All validations passed")
        
        system_prompt = f"""
Create a FINAL, production-ready architecture document.

**Problem**: {state['user_problem']}
**Iterations**: {state['iteration_count']}
**Validation**: {validation[:300]}...

**Components** (already validated):
{components}

**Architecture Overview**:
{proposed.get('architecture_summary', '')[:500]}...

Create a comprehensive document with:
1. Executive Summary
2. Architecture Overview (text diagram)
3. Detailed Component Specifications
4. Integration Points
5. Security Architecture
6. Cost Optimization
7. Deployment Plan
8. Operations & Monitoring

This is ready for implementation.
        """
        
        messages = [SystemMessage(content=system_prompt)]
        reasoning_llm = llm_manager.get_reasoning_llm()
        response = reasoning_llm.invoke(messages)
        
        final_doc = getattr(response, "content", "Final document unavailable")
        
        duration = time.time() - start_time
        logger.info(f"Final architecture document created in {duration:.2f}s")
        
        return {
            "final_architecture": {
                "document": final_doc,
                "components": components,
                "validation_summary": validation,
                "iterations": state['iteration_count']
            },
            "architecture_summary": final_doc
        }
    
    except Exception as e:
        logger.error(f"Final generation error: {e}")
        return {
            "final_architecture": {
                "document": f"Error: {str(e)}",
                "error": str(e)
            }
        }