# ============================================================================
# FILE: nodes/architects.py
# PURPOSE: Domain-specific architect implementations
# ============================================================================

from typing import cast, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.types import ArchitectureState
from core.execution import execute_tool_calls
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# WHY SEPARATE ARCHITECT FOR EACH DOMAIN?
# ============================================================================
# Each domain has different expertise, considerations, and tools:
# - COMPUTE: EC2 sizing, Lambda concurrency, Auto Scaling policies
# - NETWORK: VPC CIDR planning, Security Groups, Load Balancing
# - STORAGE: S3 tiers, EBS optimization, Backup strategies
# - DATABASE: Multi-AZ setup, Read replicas, Connection pooling
#
# While the code structure is similar, the prompts are very different.
# Keeping them separate makes it easy to customize each one.
# ============================================================================


def format_component_recommendations(
    domain_name: str,
    task_info: Dict[str, Any],
    generated_text: Optional[str]
) -> str:
    """
    Format architect output or fallback when generation fails.
    
    WHY? Graceful degradation - if LLM fails, we still return something useful.
    """
    if generated_text and generated_text.strip():
        return generated_text.strip()
    
    # Fallback to structured format
    requirements = task_info.get("requirements", []) or []
    deliverables = task_info.get("deliverables", []) or []
    
    sections = [
        f"### {domain_name.capitalize()} Domain Architecture",
        f"**Task**: {task_info.get('task_description', 'N/A')}",
    ]
    
    if requirements:
        sections.append("\n**Requirements:**")
        sections.extend(f"- {req}" for req in requirements)
    
    if deliverables:
        sections.append("\n**Deliverables:**")
        sections.extend(f"- {deliv}" for deliv in deliverables)
    
    sections.append("\n*(Generated content unavailable)*")
    return "\n".join(sections)


def generic_domain_architect(
    state: ArchitectureState,
    domain: str,
    domain_services: str,
    llm_manager,
    tool_manager,
    timeout: float = 120.0
) -> ArchitectureState:
    """
    Generic architect function for any domain.
    
    WHY GENERIC?
    Instead of duplicating code 4 times (compute, network, storage, database),
    we parameterize the domain and services. DRY principle.
    
    FLOW:
    1. Get task assigned to this domain from state
    2. Get validation feedback from previous iteration (if any)
    3. Create system prompt specific to this domain
    4. Execute LLM with tools (web search + RAG)
    5. Return architecture recommendations
    
    Args:
        state: Current state
        domain: "compute", "network", "storage", or "database"
        domain_services: Description of services in this domain
        llm_manager: Manager for LLM instances
        tool_manager: Manager for tools (web search, RAG)
        timeout: Max seconds for tool execution
    
    Returns:
        Updated state with architecture_components
    
    Example:
        result = generic_domain_architect(
            state,
            domain="compute",
            domain_services="EC2, Lambda, ECS, EKS, Auto Scaling",
            llm_manager=llm_manager,
            tool_manager=tool_manager
        )
    """
    
    node_name = f"{domain}_architect"
    logger.info(f"--- {domain.capitalize()} Architect ---")
    start_time = time.time()
    
    try:
        # ============ GET TASK FOR THIS DOMAIN ============
        domain_task = state["architecture_domain_tasks"].get(domain, {})
        
        if not domain_task or not domain_task.get("task_description"):
            error_msg = f"No task assigned for {domain} domain"
            logger.warning(error_msg)
            return cast(ArchitectureState, {
                "architecture_components": {
                    domain: {
                        "recommendations": error_msg,
                        "task_info": {},
                        "error": "No task"
                    }
                }
            })
        
        # ============ GET PREVIOUS VALIDATION FEEDBACK ============
        # If this is not the first iteration, we want to incorporate feedback
        validation_feedback = state.get("validation_feedback", [])
        domain_feedback = [
            fb for fb in validation_feedback
            if isinstance(fb, dict) and fb.get("domain", "").lower() == domain.lower()
        ]
        
        feedback_context = ""
        if domain_feedback:
            feedback_context = "\n\n**Issues Found in Previous Validation:**\n"
            for fb in domain_feedback:
                has_errors = fb.get("has_errors", False)
                status = "❌ ERRORS" if has_errors else "✓ PASSED"
                result = fb.get("validation_result", "")[:200]
                feedback_context += f"{status}: {result}...\n"
        
        # ============ CREATE SYSTEM PROMPT ============
        # This is domain-specific. Different prompts for compute vs network.
        overall_goals = state["architecture_domain_tasks"].get("overall_goals", [])
        constraints = state["architecture_domain_tasks"].get("constraints", [])
        
        system_prompt = f"""
You are an AWS {domain.capitalize()} Domain Architect.
Your expertise: {domain_services}

**Original Problem**: {state["user_problem"]}
**Iteration**: {state["iteration_count"]}/{state["max_iterations"]}

**Your Task**:
- Description: {domain_task.get('task_description', 'Design infrastructure')}
- Requirements: {', '.join(domain_task.get('requirements', []))}
- Deliverables: {', '.join(domain_task.get('deliverables', []))}

**Overall Architecture Goals**: {', '.join(overall_goals)}
**Global Constraints**: {', '.join(constraints)}

{feedback_context}

**What You Should Do**:
1. Design {domain} infrastructure for the problem
2. Use web_search for current best practices if needed
3. Use RAG_search to validate against AWS documentation
4. Provide detailed, production-ready recommendations
5. Focus ONLY on {domain} - other architects handle other domains

**If Refining**:
If this is not the first iteration and feedback was provided,
address the issues that were found. Explain your improvements.
        """
        
        # ============ PREPARE MESSAGES ============
        # Messages are local to this function - NOT added to global state
        # Why? Prevent exponential message growth across iterations
        local_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_problem"])
        ]
        
        # ============ EXECUTE WITH TOOLS ============
        tools_dict = tool_manager.get_all_tools()
        llm_with_tools = llm_manager.get_mini_with_tools(list(tools_dict.values()))
        
        final_response = execute_tool_calls(
            local_messages,
            llm_with_tools,
            tools_dict,
            max_iterations=5,
            timeout=timeout,
            retry_attempts=2
        )
        
        # ============ VALIDATE RESPONSE ============
        if not final_response:
            raise ValueError("No response from LLM")
        
        content = getattr(final_response, "content", "")
        if not content or not content.strip():
            raise ValueError("Empty response from LLM")
        
        recommendations = format_component_recommendations(domain, domain_task, content)
        
        duration = time.time() - start_time
        logger.info(f"{domain.capitalize()} architect completed in {duration:.2f}s")
        
        # ============ RETURN UPDATE ============
        return cast(ArchitectureState, {
            "architecture_components": {
                domain: {
                    "recommendations": recommendations,
                    "domain": domain,
                    "task_info": domain_task
                }
            }
        })
    
    except Exception as e:
        error_msg = f"Error in {domain} architect: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return cast(ArchitectureState, {
            "architecture_components": {
                domain: {
                    "recommendations": error_msg,
                    "domain": domain,
                    "error": str(e)
                }
            }
        })


# ============================================================================
# CONCRETE ARCHITECT FUNCTIONS
# ============================================================================
# These are thin wrappers around the generic function with domain-specific params.
# This approach is DRY - all the logic is in generic_domain_architect.
# ============================================================================

def compute_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """
    Architect for compute domain: EC2, Lambda, ECS, EKS, Auto Scaling, etc.
    """
    return generic_domain_architect(
        state,
        domain="compute",
        domain_services="EC2, Lambda, ECS, EKS, Auto Scaling, ElastiCache, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )


def network_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """
    Architect for network domain: VPC, ALB, Route 53, CloudFront, Security Groups, etc.
    """
    return generic_domain_architect(
        state,
        domain="network",
        domain_services="VPC, Subnets, Security Groups, NACLs, ALB, NLB, CloudFront, Route 53, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )


def storage_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """
    Architect for storage domain: S3, EBS, EFS, Glacier, etc.
    """
    return generic_domain_architect(
        state,
        domain="storage",
        domain_services="S3, EBS, EFS, Glacier, AWS Backup, Storage Gateway, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )


def database_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """
    Architect for database domain: RDS, DynamoDB, ElastiCache, etc.
    """
    return generic_domain_architect(
        state,
        domain="database",
        domain_services="RDS (MySQL, PostgreSQL), DynamoDB, ElastiCache, Aurora, DocumentDB, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )