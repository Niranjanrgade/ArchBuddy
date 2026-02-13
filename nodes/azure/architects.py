# ============================================================================
# FILE: nodes/azure/architects.py
# PURPOSE: Azure-specific architect implementations
# ============================================================================

from typing import cast, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from core.types import ArchitectureState
from core.execution import execute_tool_calls
import logging
import time

logger = logging.getLogger(__name__)


def format_component_recommendations(
    domain_name: str,
    task_info: Dict[str, Any],
    generated_text: Optional[str]
) -> str:
    """Format architect output or fallback when generation fails."""
    if generated_text and generated_text.strip():
        return generated_text.strip()
    
    requirements = task_info.get("requirements", []) or []
    deliverables = task_info.get("deliverables", []) or []
    
    sections = [
        f"### {domain_name.capitalize()} Domain Architecture (Azure)",
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
    
    ROLE: Generate architecture recommendations for a domain.
    INPUT: Task from supervisor, validation feedback (if any)
    OUTPUT: Architecture recommendations for this domain
    """
    
    node_name = f"azure_{domain}_architect"
    logger.info(f"--- Azure {domain.capitalize()} Architect ---")
    start_time = time.time()
    
    try:
        # Get task for this domain
        domain_task = state["architecture_domain_tasks"].get(domain, {})
        
        if not domain_task or not domain_task.get("task_description"):
            # It's possible the task key is generic, e.g. "compute", but we are the Azure architect.
            # State structure for tasks is { "compute": ... }. This is shared unless we separate tasks by provider.
            # For now, we assume the supervisor generates generic tasks, or we update supervisor to generate provider-specific tasks?
            # Plan said: "Update nodes/supervisor.py to support multiple providers".
            # If supervisor generates "compute" task, both AWS and Azure architects can pick it up.
            
            error_msg = f"No task assigned for {domain} domain"
            logger.warning(error_msg)
            return cast(ArchitectureState, {
                "architecture_components": {
                    f"azure_{domain}": { # Use provider prefix to avoid collision if parallel
                        "recommendations": error_msg,
                        "task_info": {},
                        "error": "No task"
                    }
                }
            })
        
        # Get previous validation feedback
        validation_feedback = state.get("validation_feedback", [])
        domain_feedback = [
            fb for fb in validation_feedback
            if isinstance(fb, dict) and fb.get("domain", "").lower() == f"azure_{domain}".lower()
        ]
        
        feedback_context = ""
        if domain_feedback:
            feedback_context = "\n\n**Issues Found in Previous Validation:**\n"
            for fb in domain_feedback:
                has_errors = fb.get("has_errors", False)
                status = "❌ ERRORS" if has_errors else "✓ PASSED"
                result = fb.get("result", "")[:200]
                feedback_context += f"{status}: {result}...\n"
        
        # Create system prompt
        overall_goals = state.get("architecture_domain_tasks", {}).get("overall_goals", [])
        constraints = state.get("architecture_domain_tasks", {}).get("constraints", [])
        
        system_prompt = f"""
You are an Azure {domain.capitalize()} Domain Architect.
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
1. Design Azure infrastructure for the problem
2. Use web_search for current best practices if needed
3. Use RAG_search to validate against documentation (if available) or web_search
4. Provide detailed, production-ready recommendations using Azure services
5. Focus ONLY on {domain}

**If Refining**:
If this is not the first iteration and feedback was provided,
address the issues that were found. Explain your improvements.
        """
        
        # Prepare messages
        local_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_problem"])
        ]
        
        # Execute with tools
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
        
        # Validate response
        if not final_response:
            raise ValueError("No response from LLM")
        
        content = getattr(final_response, "content", "")
        if not content or not content.strip():
            raise ValueError("Empty response from LLM")
        
        recommendations = format_component_recommendations(domain, domain_task, content)
        
        duration = time.time() - start_time
        logger.info(f"Azure {domain.capitalize()} architect completed in {duration:.2f}s")
        
        # Store with provider prefix or structural change?
        # If we run parallel, we need to distinguish AWS vs Azure content.
        # I will use "azure_<domain>" as key in architecture_components if keys are flexible.
        # ArchitectureState definition: architecture_components: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
        # So keys can be anything.
        
        key = f"azure_{domain}"
        
        return cast(ArchitectureState, {
            "architecture_components": {
                key: {
                    "recommendations": recommendations,
                    "domain": domain,
                    "provider": "azure",
                    "task_info": domain_task
                }
            }
        })
    
    except Exception as e:
        error_msg = f"Error in Azure {domain} architect: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return cast(ArchitectureState, {
            "architecture_components": {
                f"azure_{domain}": {
                    "recommendations": error_msg,
                    "domain": domain,
                    "provider": "azure",
                    "error": str(e)
                }
            }
        })


def azure_compute_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Architect for Azure compute domain."""
    return generic_domain_architect(
        state,
        domain="compute",
        domain_services="Virtual Machines, Azure Functions, AKS (Kubernetes), App Service, Azure Container Apps, Batch, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )


def azure_network_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Architect for Azure network domain."""
    return generic_domain_architect(
        state,
        domain="network",
        domain_services="Virtual Network (VNet), NSG, Application Gateway, Azure Front Door, ExpressRoute, VPN Gateway, DNS, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )


def azure_storage_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Architect for Azure storage domain."""
    return generic_domain_architect(
        state,
        domain="storage",
        domain_services="Blob Storage, Azure Files, Disk Storage, Data Lake Storage, Azure NetApp Files, Backup, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )


def azure_database_architect(state: ArchitectureState, llm_manager, tool_manager) -> ArchitectureState:
    """Architect for Azure database domain."""
    return generic_domain_architect(
        state,
        domain="database",
        domain_services="Azure SQL Database, Cosmos DB, Azure Database for PostgreSQL/MySQL, Redis Cache, etc.",
        llm_manager=llm_manager,
        tool_manager=tool_manager
    )
