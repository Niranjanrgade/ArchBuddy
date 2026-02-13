# ============================================================================
# FILE: nodes/azure/validators.py
# PURPOSE: Validate architecture against Azure documentation
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
    """Services to validate for each domain (Azure)."""
    services = {
        "compute": ["Virtual Machines", "Azure Functions", "AKS", "App Service", "Container Apps"],
        "network": ["VNet", "NSG", "Application Gateway", "Front Door", "ExpressRoute"],
        "storage": ["Blob Storage", "Azure Files", "Disk Storage", "Data Lake"],
        "database": ["Azure SQL", "Cosmos DB", "PostgreSQL", "Redis"]
    }
    return services.get(domain, [])


def get_validation_focus(domain: str) -> str:
    """What to specifically check (Azure)."""
    focus = {
        "compute": "VM SKUs, scaling (VMSS), availability zones, cost tiers",
        "network": "Subnet layout, NSG rules, load balancing options, private endpoints",
        "storage": "Replication (LRS/GRS), access tiers, private link, security",
        "database": "DTU vs vCore, consistency levels, backup retention, HA"
    }
    return focus.get(domain, "general validation")


def azure_validator_supervisor(
    state: ArchitectureState,
    llm_manager
) -> ArchitectureState:
    """
    Decide what to validate for Azure components.
    
    ROLE: Create validation tasks for each domain.
    INPUT: Proposed architecture from synthesizer
    OUTPUT: Validation tasks assigned to validators
    """
    
    logger.info("--- Azure Validator Supervisor ---")
    
    try:
        architecture_components = state.get("architecture_components", {})
        
        # Filter for Azure components
        # We assume Azure components keys start with 'azure_' based on architects.py
        azure_keys = [k for k in architecture_components.keys() if k.startswith("azure_")]
        
        validation_tasks_update = {}
        
        for key in azure_keys:
            # key is like "azure_compute"
            domain = key.replace("azure_", "")
            validation_tasks_update[key] = {
                "components_to_validate": get_services_for_domain(domain),
                "validation_focus": get_validation_focus(domain)
            }
        
        if not validation_tasks_update:
            logger.info("No Azure components to validate")
            return cast(ArchitectureState, {
                "architecture_domain_tasks": state.get("architecture_domain_tasks", {})
            })
        
        # Merge with existing tasks
        from core.types import merge_dicts
        existing_tasks = state.get("architecture_domain_tasks", {})
        merged = merge_dicts(existing_tasks, {"validation_tasks": validation_tasks_update})
        
        return cast(ArchitectureState, {
            "architecture_domain_tasks": merged
        })
    
    except Exception as e:
        logger.error(f"Azure Validator supervisor error: {e}")
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
    Validate Azure architect recommendations.
    """
    
    node_name = f"azure_{domain}_validator"
    logger.info(f"--- Azure {domain.capitalize()} Validator ---")
    start_time = time.time()
    
    try:
        # Get what architect generated
        # Key in existing state for Azure is likely "azure_compute" etc.
        component_key = f"azure_{domain}"
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
You are validating Azure architecture recommendations against official Azure documentation.

**Domain**: {domain.capitalize()}
**What to Check**: {get_validation_focus(domain)}

**Architect's Recommendation**:
{domain_recommendations}

**Your Task**:
1. Search web/docs for each service mentioned using web_search
2. Check if the configuration is valid
3. Check if best practices are followed (Well-Architected Framework)
4. List any errors or issues found
5. Rate confidence (0-100%)

DO NOT try to improve or regenerate.
ONLY report what is wrong (or correct).
        """
        
        local_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Validate {domain} recommendations")
        ]
        
        # Execute with tools - Web search primarily if RAG is AWS-focused
        tools_dict = tool_manager.get_all_tools()
        # Using all tools including web_search
        
        llm_with_tools = llm_manager.get_mini_with_tools(list(tools_dict.values()))
        
        response = execute_tool_calls(
            local_messages,
            llm_with_tools,
            tools_dict,
            timeout=timeout
        )
        
        validation_result = getattr(response, "content", "Validation completed")
        has_errors = detect_errors_llm(validation_result)
        
        duration = time.time() - start_time
        logger.info(f"Azure {domain.capitalize()} validator: {duration:.2f}s (errors: {has_errors})")
        
        return cast(ArchitectureState, {
            "validation_feedback": [{
                "domain": component_key,
                "result": validation_result,
                "has_errors": has_errors
            }],
            "has_validation_errors": has_errors
        })
    
    except Exception as e:
        logger.error(f"Azure {domain} validator error: {e}")
        return cast(ArchitectureState, {
            "validation_feedback": [{
                "domain": f"azure_{domain}",
                "result": f"Validation error: {str(e)}",
                "has_errors": True
            }],
            "has_validation_errors": True
        })


# Concrete validators
def azure_compute_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    return generic_domain_validator(state, "compute", llm_manager, tool_manager)

def azure_network_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    return generic_domain_validator(state, "network", llm_manager, tool_manager)

def azure_storage_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    return generic_domain_validator(state, "storage", llm_manager, tool_manager)

def azure_database_validator(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    return generic_domain_validator(state, "database", llm_manager, tool_manager)
