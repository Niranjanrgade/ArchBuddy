# ============================================================================
# FILE: nodes/validators.py
# PURPOSE: Validate architecture against AWS documentation
# ============================================================================

from typing import cast, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from core.types import ArchitectureState
from core.execution import execute_tool_calls, detect_errors_llm
from core.schemas import ValidationTask, ValidationDecomposition
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION STRATEGY
# ============================================================================
# After architects design the architecture, validators check if it's correct.
#
# WHAT WE VALIDATE:
# 1. Factual correctness: "Does this service exist?" "Is this config valid?"
# 2. Best practices: "Is this following AWS recommendations?"
# 3. Compatibility: "Do these services work together?"
# 4. Cost: "Is this cost-effective?"
# 5. Security: "Are there security issues?"
#
# HOW WE VALIDATE:
# - Use RAG search to find official AWS documentation
# - Use LLM to compare architect's recommendations to docs
# - Flag discrepancies as errors
#
# WHY NOT JUST ONE VALIDATOR?
# Because we need expertise per domain. A compute validator knows EC2, Lambda.
# A network validator knows VPCs, Security Groups. They're specialized.
# ============================================================================


def validator_supervisor(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Break down validation tasks by domain.
    
    WHY A SUPERVISOR?
    Just like architects need a supervisor to assign tasks,
    validators need a supervisor to decide what to validate.
    
    WHAT IT DOES:
    1. Looks at the proposed architecture
    2. Decides what needs validation in each domain
    3. Creates validation tasks
    4. Passes them to domain validators
    
    Example:
    Input: proposed_architecture mentions "EC2 t3.large auto-scaling"
    Output:
      - Validate compute: EC2, Auto Scaling config
      - Validate network: VPC, Security Groups for EC2
      - Validate storage: Any EBS volumes for EC2
    """
    
    logger.info("--- Validator Supervisor ---")
    
    try:
        architecture_components = state.get("architecture_components", {})
        proposed_architecture = state.get("proposed_architecture", {})
        
        system_prompt = f"""
You are a validation supervisor for AWS architecture.
You will break down the architecture into validation tasks.

**Architecture Components**:
{architecture_components}

**Proposed Architecture**:
{proposed_architecture.get('architecture_summary', 'N/A')[:5000]}...

For each domain with components (compute, network, storage, database):
1. List specific AWS services to validate
2. Specify validation focus (config, best practices, compatibility)
3. Explain what to check

Output as JSON matching ValidationDecomposition schema.
        """
        
        try:
            # Get structured LLM
            structured_llm = llm_manager.get_reasoning_structured(ValidationDecomposition)
            messages = [SystemMessage(content=system_prompt)]
            
            response = structured_llm.invoke(messages)
            validation_decomposition = cast(ValidationDecomposition, response)
            
            if not validation_decomposition or not validation_decomposition.validation_tasks:
                raise ValueError("Empty validation decomposition")
        
        except Exception as e:
            logger.warning(f"Structured output failed, returning empty tasks: {e}")
            validation_decomposition = ValidationDecomposition(validation_tasks=[])
        
        # ============ FORMAT FOR STATE ============
        validation_tasks_update = {}
        for task in validation_decomposition.validation_tasks:
            domain_key = task.domain.lower()
            validation_tasks_update[domain_key] = {
                "components_to_validate": task.components_to_validate,
                "validation_focus": task.validation_focus
            }
        
        # Merge with existing domain tasks
        from core.types import merge_dicts
        existing_tasks = state.get("architecture_domain_tasks", {})
        merged = merge_dicts(existing_tasks, {"validation_tasks": validation_tasks_update})
        
        logger.info(f"Created {len(validation_tasks_update)} validation tasks")
        
        return cast(ArchitectureState, {
            "architecture_domain_tasks": merged
        })
    
    except Exception as e:
        logger.error(f"Validator supervisor error: {e}", exc_info=True)
        return cast(ArchitectureState, {
            "architecture_domain_tasks": state.get("architecture_domain_tasks", {})
        })


def generic_domain_validator(
    state: ArchitectureState,
    domain: str,
    validation_focus_description: str,
    llm_manager,
    tool_manager,
    timeout: float = 300.0
) -> ArchitectureState:
    """
    Generic validator for any domain.
    
    FLOW:
    1. Get validation task for this domain
    2. Get the architect's recommendations for this domain
    3. Use RAG to search AWS docs for the services mentioned
    4. Compare recommendations to documentation
    5. Flag any errors or issues
    6. Return feedback
    
    Args:
        state: Current state
        domain: "compute", "network", "storage", or "database"
        validation_focus_description: What to specifically validate
        llm_manager: LLM manager
        tool_manager: Tool manager (for RAG)
    
    Returns:
        Updated state with validation_feedback
    """
    
    node_name = f"{domain}_validator"
    logger.info(f"--- {domain.capitalize()} Validator ---")
    start_time = time.time()
    
    try:
        # ============ GET VALIDATION TASK ============
        validation_tasks = state.get("architecture_domain_tasks", {}).get("validation_tasks", {})
        domain_validation = validation_tasks.get(domain, {})
        
        if not domain_validation:
            logger.info(f"No validation task for {domain}, skipping")
            return cast(ArchitectureState, {
                "validation_feedback": [{
                    "domain": domain,
                    "status": "skipped",
                    "validation_result": f"No validation tasks for {domain}",
                    "components_validated": [],
                    "has_errors": False
                }]
            })
        
        # ============ GET RECOMMENDATIONS TO VALIDATE ============
        components = state.get("architecture_components", {})
        domain_components = components.get(domain, {})
        recommendations = domain_components.get('recommendations', '')
        
        # ============ CREATE VALIDATION PROMPT ============
        components_to_validate = domain_validation.get("components_to_validate", [])
        validation_focus = domain_validation.get("validation_focus", "general validation")
        
        system_prompt = f"""
You are a {domain.capitalize()} Domain Validator for AWS.
Validate the architecture against AWS documentation.

**Components to Validate**: {', '.join(components_to_validate)}
**Validation Focus**: {validation_focus}
**What to Check**:
{validation_focus_description}

**Proposed {domain.capitalize()} Architecture**:
{recommendations}

**How to Validate**:
1. Use RAG_search to find AWS documentation for each service
2. Check if the recommendations match the docs
3. Flag any errors, misconfigurations, or missing best practices
4. Rate your confidence level

**Report Format**:
- List valid components (correctly configured)
- List issues (errors, gaps, improvements)
- Recommendations for fixes
- Overall confidence (0-100%)
        """
        
        local_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Validate these {domain} components: {', '.join(components_to_validate)}")
        ]
        
        # ============ EXECUTE WITH TOOLS ============
        tools_dict = tool_manager.get_all_tools()
        rag_tools = {k: v for k, v in tools_dict.items() if k == "RAG_search"}
        
        llm_with_tools = llm_manager.get_mini_with_tools(list(rag_tools.values()))
        
        final_response = execute_tool_calls(
            local_messages,
            llm_with_tools,
            rag_tools,
            timeout=timeout
        )
        
        validation_result = getattr(final_response, "content", "Validation completed")
        
        # ============ DETECT ERRORS ============
        # Use LLM to intelligently detect if there are errors
        has_errors = detect_errors_llm(validation_result)
        
        duration = time.time() - start_time
        logger.info(f"{domain.capitalize()} validator completed in {duration:.2f}s (errors: {has_errors})")
        
        # ============ RETURN FEEDBACK ============
        return cast(ArchitectureState, {
            "validation_feedback": [{
                "domain": domain,
                "validation_result": validation_result,
                "components_validated": components_to_validate,
                "has_errors": has_errors
            }],
            "factual_errors_exist": has_errors
        })
    
    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return cast(ArchitectureState, {
            "validation_feedback": [{
                "domain": domain,
                "validation_result": error_msg,
                "components_validated": [],
                "has_errors": True
            }],
            "factual_errors_exist": True
        })


# ============================================================================
# CONCRETE VALIDATOR FUNCTIONS
# ============================================================================

def compute_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate compute domain architecture."""
    validation_focus = """
1. Service names exist and versions are correct
2. Instance types and sizing are appropriate
3. Auto Scaling configurations are valid
4. Best practices are followed
5. Configuration parameters are valid
    """
    return generic_domain_validator(
        state, "compute", validation_focus, llm_manager, tool_manager
    )


def network_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate network domain architecture."""
    validation_focus = """
1. VPC CIDR blocks and subnet sizing
2. Security Group rules are valid
3. Load Balancer configurations
4. DNS and CDN setup
5. Network connectivity flow
    """
    return generic_domain_validator(
        state, "network", validation_focus, llm_manager, tool_manager
    )


def storage_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate storage domain architecture."""
    validation_focus = """
1. S3 bucket configuration and access
2. EBS volume types and configurations
3. EFS setup and performance
4. Lifecycle policies
5. Encryption and compliance
    """
    return generic_domain_validator(
        state, "storage", validation_focus, llm_manager, tool_manager
    )


def database_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Validate database domain architecture."""
    validation_focus = """
1. Engine selection and versions
2. Instance sizing and types
3. Backup and recovery settings
4. High availability setup
5. Security and encryption
    """
    return generic_domain_validator(
        state, "database", validation_focus, llm_manager, tool_manager
    )


def validation_synthesizer(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Summarize all validation feedback into actionable insights.
    
    WHY SEPARATE FROM VALIDATORS?
    Validators run in parallel and each produces domain-specific feedback.
    Synthesizer runs AFTER all validators complete and creates a unified summary.
    This summary informs the iteration decision.
    """
    
    logger.info("--- Validation Synthesizer ---")
    
    try:
        all_feedback = state.get("validation_feedback", [])
        
        if not all_feedback:
            return cast(ArchitectureState, {
                "validation_summary": "No validation feedback available"
            })
        
        # ============ PREPARE FEEDBACK FOR SUMMARY ============
        feedback_summaries = []
        error_count = 0
        
        for feedback in all_feedback:
            domain = feedback.get("domain", "unknown")
            has_errors = feedback.get("has_errors", False)
            result = feedback.get("validation_result", "")[:300]
            
            if has_errors:
                error_count += 1
            
            feedback_summaries.append(f"""
**{domain.upper()}**: {result}...
            """)
        
        # ============ CREATE SUMMARY PROMPT ============
        system_prompt = f"""
You are synthesizing validation feedback from 4 domain validators.

**Validation Feedback**:
{''.join(feedback_summaries)}

**Error Count**: {error_count} domains have issues

Create a concise summary that includes:
1. Overall validation status (passed/failed)
2. Critical issues that must be fixed
3. Non-critical improvements
4. Recommendations for next iteration (if needed)
        """
        
        messages = [SystemMessage(content=system_prompt)]
        reasoning_llm = llm_manager.get_reasoning_llm()
        response = reasoning_llm.invoke(messages)
        
        if not response or not hasattr(response, "content"):
            raise ValueError("Empty response from validation synthesizer")
        
        validation_summary = response.content
        
        logger.info("Validation synthesizer completed")
        
        return cast(ArchitectureState, {
            "validation_summary": validation_summary
        })
    
    except Exception as e:
        logger.error(f"Validation synthesizer error: {e}", exc_info=True)
        return cast(ArchitectureState, {
            "validation_summary": f"Error synthesizing feedback: {str(e)}"
        })