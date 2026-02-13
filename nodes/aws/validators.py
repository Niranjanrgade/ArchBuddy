# ============================================================================
# FILE: nodes/validators.py
# PURPOSE: Validate architecture against AWS documentation
# ============================================================================

from typing import cast, Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from core.types import ArchitectureState
from core.execution import execute_tool_calls, detect_errors_llm
from core.schemas import ValidationDecomposition
import logging
import time

logger = logging.getLogger(__name__)


def get_services_for_domain(domain: str) -> List[str]:
    """Services to validate for each domain."""
    services = {
        "compute": ["EC2", "Lambda", "ECS", "EKS", "Auto Scaling"],
        "network": ["VPC", "Security Groups", "ALB", "NLB", "Route 53"],
        "storage": ["S3", "EBS", "EFS", "Glacier"],
        "database": ["RDS", "DynamoDB", "ElastiCache", "Aurora"]
    }
    return services.get(domain, [])


def get_validation_focus(domain: str) -> str:
    """What to specifically check."""
    focus = {
        "compute": "Instance types, sizing, Auto Scaling config, cost optimization",
        "network": "VPC design, security groups, load balancing, routing",
        "storage": "Bucket policies, encryption, lifecycle, access controls",
        "database": "Engine choice, replication, backup, high availability"
    }
    return focus.get(domain, "general validation")


def validator_supervisor(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Decide what to validate.
    
    ROLE: Create validation tasks for each domain.
    INPUT: Proposed architecture from synthesizer
    OUTPUT: Validation tasks assigned to validators
    """
    
    logger.info("--- Validator Supervisor ---")
    
    try:
        architecture_components = state.get("architecture_components", {})
        
        # For each domain that has components, create validation task
        validation_tasks_update = {}
        
        # Filter for AWS components
        aws_keys = [k for k in architecture_components.keys() if k.startswith("aws_")]
        
        for key in aws_keys:
            # key is "aws_compute", domain_type is "compute"
            domain_type = key.replace("aws_", "")
            validation_tasks_update[key] = {
                "components_to_validate": get_services_for_domain(domain_type),
                "validation_focus": get_validation_focus(domain_type)
            }
        
        if not validation_tasks_update:
            logger.info("No components to validate")
            return cast(ArchitectureState, {
                "architecture_domain_tasks": state.get("architecture_domain_tasks", {})
            })
        
        # Merge with existing tasks
        from core.types import merge_dicts
        existing_tasks = state.get("architecture_domain_tasks", {})
        merged = merge_dicts(existing_tasks, {"validation_tasks": validation_tasks_update})
        
        logger.info(f"Created {len(validation_tasks_update)} validation tasks")
        
        return cast(ArchitectureState, {
            "architecture_domain_tasks": merged
        })
    
    except Exception as e:
        logger.error(f"Validator supervisor error: {e}")
        return cast(ArchitectureState, {
            "architecture_domain_tasks": state.get("architecture_domain_tasks", {})
        })


def generic_domain_validator(
    state: ArchitectureState,
    domain: str,
    llm_manager,
    tool_manager,
    timeout: float = 60.0
) -> ArchitectureState:
    """
    Validate architect recommendations against AWS docs.
    
    ROLE: Check if recommendations are correct.
    INPUT: Architect's recommendations
    OUTPUT: Validation feedback (pass/fail with details)
    """
    
    node_name = f"{domain}_validator"
    logger.info(f"--- {domain.capitalize()} Validator ---")
    start_time = time.time()
    
    try:
        # Get what architect generated
        component_key = f"aws_{domain}"
        components = state.get("architecture_components", {})
        domain_recommendations = components.get(component_key, {}).get("recommendations", "")
        
        if not domain_recommendations:
            return cast(ArchitectureState, {
                "validation_feedback": [{
                    "domain": component_key,
                    "status": "skipped",
                    "result": "No recommendations to validate",
                    "has_errors": False
                }]
            })
        
        # Create validation prompt
        system_prompt = f"""
You are validating AWS architecture recommendations against official AWS documentation.

**Domain**: {domain.capitalize()}
**What to Check**: {get_validation_focus(domain)}

**Architect's Recommendation**:
{domain_recommendations}

**Your Task**:
1. Search AWS docs for each service mentioned using RAG_search
2. Check if the configuration is valid
3. Check if best practices are followed
4. List any errors or issues found
5. Rate confidence (0-100%)

DO NOT try to improve or regenerate.
ONLY report what is wrong (or correct).

**Report Format**:
- Valid components (if any)
- Issues found (be specific)
- Severity (critical/warning/info)
- Confidence in assessment
        """
        
        local_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Validate {domain} recommendations")
        ]
        
        # Execute with tools - ONLY RAG for documentation search
        tools_dict = tool_manager.get_all_tools()
        rag_only = {"RAG_search": tools_dict["RAG_search"]}
        
        llm_with_tools = llm_manager.get_mini_with_tools(list(rag_only.values()))
        
        response = execute_tool_calls(
            local_messages,
            llm_with_tools,
            rag_only,
            timeout=timeout
        )
        
        validation_result = getattr(response, "content", "Validation completed")
        has_errors = detect_errors_llm(validation_result)
        
        duration = time.time() - start_time
        logger.info(f"{domain.capitalize()} validator: {duration:.2f}s (errors: {has_errors})")
        
        return cast(ArchitectureState, {
            "validation_feedback": [{
                "domain": domain,
                "result": validation_result,
                "has_errors": has_errors
            }],
            "has_validation_errors": has_errors
        })
    
    except Exception as e:
        logger.error(f"{domain} validator error: {e}")
        return cast(ArchitectureState, {
            "validation_feedback": [{
                "domain": domain,
                "result": f"Validation error: {str(e)}",
                "has_errors": True
            }],
            "has_validation_errors": True
        })


# Concrete validators
def compute_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate compute domain."""
    return generic_domain_validator(state, "compute", llm_manager, tool_manager)


def network_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate network domain."""
    return generic_domain_validator(state, "network", llm_manager, tool_manager)


def storage_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate storage domain."""
    return generic_domain_validator(state, "storage", llm_manager, tool_manager)


def database_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate database domain."""
    return generic_domain_validator(state, "database", llm_manager, tool_manager)


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
            f"**{fb['domain'].upper()}**: {fb['result'][:200]}..."
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