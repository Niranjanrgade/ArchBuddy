# ============================================================================
# FILE: nodes/supervisor.py
# PURPOSE: Architect supervisor node
# ============================================================================

from typing import cast
from langchain_core.messages import SystemMessage
from core.types import ArchitectureState
from core.schemas import TaskDecomposition
import logging
import time

logger = logging.getLogger(__name__)


def architect_supervisor(
    state: ArchitectureState,
    llm_manager,
    max_retries: int = 3
) -> ArchitectureState:
    """
    Break down user's problem into tasks for domain architects.
    
    ROLE: Orchestrate architecture generation.
    INPUT: User's problem statement
    OUTPUT: Tasks assigned to compute, network, storage, database architects
    """
    
    iteration = state["iteration_count"] + 1
    logger.info(f"Architect Supervisor (Iteration {iteration}/{state['max_iterations']})")
    
    try:
        # Get feedback from previous iteration
        previous_feedback = state.get("validation_feedback", [])
        feedback_context = ""
        if previous_feedback:
            feedback_context = "\n\n**Issues found in previous iteration:**\n"
            for fb in previous_feedback:
                domain = fb.get("domain", "?")
                result = fb.get("result", "")[:100]
                feedback_context += f"- {domain}: {result}...\n"
        
        system_prompt = f"""
You are an AWS architect supervisor.
Break down the user's problem into tasks for different domain architects.

User Problem: {state['user_problem']}
Iteration: {iteration}/{state['max_iterations']}

{feedback_context}

Create detailed tasks for these domains:
1. Compute (EC2, Lambda, ECS, EKS)
2. Network (VPC, ALB, Route 53, CloudFront)
3. Storage (S3, EBS, EFS)
4. Database (RDS, DynamoDB, ElastiCache)

For each domain, provide:
- Clear task description
- Key requirements
- Expected deliverables
- Constraints

If this is a refinement iteration, address the issues found.
        """
        
        structured_llm = llm_manager.get_reasoning_structured(TaskDecomposition)
        messages = [SystemMessage(content=system_prompt)]
        
        task_decomposition = None
        for attempt in range(max_retries):
            try:
                response = structured_llm.invoke(messages)
                task_decomposition = cast(TaskDecomposition, response)
                
                if task_decomposition and task_decomposition.decomposed_tasks:
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        if not task_decomposition:
            raise ValueError("Could not generate task decomposition")
        
        # Format for state
        domain_tasks_update = {
            "overall_goals": task_decomposition.overall_architecture_goals,
            "constraints": task_decomposition.constraints,
        }
        
        for task in task_decomposition.decomposed_tasks:
            domain_key = task.domain.lower()
            domain_tasks_update[domain_key] = {
                "task_description": task.task_description,
                "requirements": task.requirements,
                "deliverables": task.deliverables
            }
        
        logger.info("Supervisor completed successfully")
        
        return cast(ArchitectureState, {
            "architecture_domain_tasks": domain_tasks_update,
            "iteration_count": iteration,
            "validation_feedback": [],
            "architecture_components": {},
            "has_validation_errors": False,
        })
    
    except Exception as e:
        logger.error(f"Supervisor error: {e}", exc_info=True)
        return cast(ArchitectureState, {
            "iteration_count": iteration,
            "has_validation_errors": True,
        })