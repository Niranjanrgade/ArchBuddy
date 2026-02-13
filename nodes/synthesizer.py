# ============================================================================
# FILE: nodes/synthesizer.py
# PURPOSE: Combine architecture components from all domains
# ============================================================================

from typing import cast
from langchain_core.messages import SystemMessage
from core.types import ArchitectureState
import logging
import time

logger = logging.getLogger(__name__)


def architect_synthesizer(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Synthesize individual domain architectures into a unified design.
    
    ROLE: Integration point after all architects complete.
    INPUT: All domain-specific recommendations
    OUTPUT: Coherent unified architecture
    """
    
    logger.info("--- Architect Synthesizer ---")
    start_time = time.time()
    
    try:
        all_components = state.get("architecture_components", {})
        
        if not all_components:
            logger.warning("No architecture components to synthesize")
            return cast(ArchitectureState, {
                "proposed_architecture": {
                    "architecture_summary": "No components available for synthesis"
                }
            })
        
        completed_domains = list(all_components.keys())
        logger.info(f"Synthesizing {len(completed_domains)} domains: {completed_domains}")
        
        # Prepare component summaries
        component_summaries = []
        for domain, info in all_components.items():
            recommendation = info.get('recommendations', 'N/A')
            if not recommendation.strip():
                recommendation = f"No recommendations for {domain}"
            
            component_summaries.append(f"""
**{domain.upper()} Component**:
{recommendation}
            """)
        
        # Create synthesizer prompt
        system_prompt = f"""
You are a Principal Cloud Solutions Architect.
You have received architecture designs from specialist architects.
Your job is to synthesize these into ONE coherent, integrated architecture.

**Original Problem**: {state['user_problem']}
**Overall Goals**: {state['architecture_domain_tasks'].get('overall_goals', [])}
**Constraints**: {state['architecture_domain_tasks'].get('constraints', [])}

**Architecture Components from Specialists**:
{''.join(component_summaries)}

**Your Task**:
1. Review all components
2. Identify integration points
3. Check for conflicts or incompatibilities
4. Create a UNIFIED architecture that ties everything together
5. Provide a comprehensive overview

**Output Format**:
- Start with a high-level summary
- Then detail how each component integrates
- Then provide the complete integrated design
- Include security, scalability, and cost considerations
        """
        
        messages = [SystemMessage(content=system_prompt)]
        reasoning_llm = llm_manager.get_reasoning_llm()
        response = reasoning_llm.invoke(messages)
        
        if not response or not hasattr(response, "content"):
            raise ValueError("Empty response from synthesizer LLM")
        
        architecture_summary = response.content
        
        duration = time.time() - start_time
        logger.info(f"Synthesizer completed in {duration:.2f}s")
        
        return cast(ArchitectureState, {
            "proposed_architecture": {
                "architecture_summary": architecture_summary,
                "source_components": all_components,
                "synthesis_iteration": state["iteration_count"]
            }
        })
    
    except Exception as e:
        error_msg = f"Synthesizer error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return cast(ArchitectureState, {
            "proposed_architecture": {
                "architecture_summary": error_msg,
                "error": str(e)
            }
        })


def validation_synthesizer(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Summarize validation results.
    
    ROLE: Create summary of all validation feedback.
    INPUT: Validation feedback from all validators
    OUTPUT: Summary report and pass/fail decision
    """
    
    logger.info("--- Validation Synthesizer ---")
    
    try:
        all_feedback = state.get("validation_feedback", [])
        
        if not all_feedback:
            return cast(ArchitectureState, {
                "validation_summary": "No validation performed",
                "has_validation_errors": False
            })
        
        # Count errors
        error_count = sum(1 for fb in all_feedback if fb.get("has_errors", False))
        
        feedback_text = "\n".join([
            f"**{fb.get('domain', 'unknown').upper()}**: {fb.get('result', '')[:200]}..."
            for fb in all_feedback
        ])
        
        # Create summary
        system_prompt = f"""
Summarize validation results for the architecture.

**Validation Results**:
{feedback_text}

**Errors Found**: {error_count} domain(s)

Create a concise summary:
1. Overall status (PASS / FAIL)
2. Critical issues that must be fixed
3. Non-critical improvements
4. What to fix if retrying
        """
        
        messages = [SystemMessage(content=system_prompt)]
        reasoning_llm = llm_manager.get_reasoning_llm()
        response = reasoning_llm.invoke(messages)
        
        summary = getattr(response, "content", "Validation summary unavailable")
        
        return cast(ArchitectureState, {
            "validation_summary": summary,
            "has_validation_errors": error_count > 0
        })
    
    except Exception as e:
        logger.error(f"Validation synthesizer error: {e}")
        return cast(ArchitectureState, {
            "validation_summary": f"Error: {str(e)}",
            "has_validation_errors": True
        })