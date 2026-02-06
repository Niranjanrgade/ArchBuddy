# ============================================================================
# FILE: nodes/supervisor.py
# PURPOSE: Architect supervisor node
# ============================================================================

from typing import cast
from langchain_core.messages import SystemMessage, AIMessage
from core.types import ArchitectureState
from core.schemas import TaskDecomposition
import logging

logger = logging.getLogger(__name__)


def architect_supervisor(
    state: ArchitectureState,
    llm_manager,
    max_retries: int = 3
) -> ArchitectureState:
    """
    Orchestrate architecture generation.
    
    ROLE: Break down user's problem into tasks for domain architects.
    
    WHAT HAPPENS:
    1. Takes user problem (e.g., "Build microservices on AWS")
    2. Creates a task for each domain (compute, network, storage, database)
    3. Returns these tasks for domain architects to work on
    
    HOW IT WORKS:
    - Uses structured output (Pydantic model) to force LLM to return valid tasks
    - With_structured_output ensures output matches TaskDecomposition schema
    - If LLM fails, retry up to 3 times
    
    WHY SEPARATE FROM VALIDATORS?
    - Supervisor makes architectural decisions (what to build)
    - Validators check correctness (is it correct?)
    - These are different concerns, so different nodes
    """
    
    iteration = state["iteration_count"] + 1
    logger.info(f"Architect Supervisor (Iteration {iteration}/{state['max_iterations']})")
    
    try:
        # Get feedback from previous iteration
        previous_feedback = state.get("validation_feedback", [])
        feedback_context = ""
        if previous_feedback:
            feedback_context = "\n\nIssues found in previous iteration:\n"
            for fb in previous_feedback:
                domain = fb.get("domain", "?")
                feedback_context += f"- {domain}: {fb.get('validation_result', '')[:100]}...\n"
        
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
        
        # Get LLM with structured output
        structured_llm = llm_manager.get_reasoning_structured(TaskDecomposition)
        
        # Retry loop
        task_decomposition = None
        for attempt in range(max_retries):
            try:
                messages = [SystemMessage(content=system_prompt)]
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
            "validation_feedback": [],  # Reset for new iteration
            "architecture_components": {},  # Clear old components
            "factual_errors_exist": False,  # Reset error flag
        })
    
    except Exception as e:
        logger.error(f"Supervisor error: {e}", exc_info=True)
        return cast(ArchitectureState, {
            "iteration_count": iteration,
            "factual_errors_exist": True,  # Mark as error
        })