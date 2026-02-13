# ============================================================================
# FILE: graph/builder.py
# PURPOSE: Construct the LangGraph workflow (Multi-Provider Support)
# ============================================================================

from langgraph.graph import StateGraph, START, END
# Note: MemorySaver is now successfully used in main.py, so we return the builder here.
from core.types import ArchitectureState
from nodes.supervisor import architect_supervisor
from nodes.synthesizer import architect_synthesizer, validation_synthesizer
from nodes.iteration import iteration_condition, final_architecture_generator

# AWS Nodes
from nodes.aws.architects import (
    compute_architect as aws_compute_architect,
    network_architect as aws_network_architect,
    storage_architect as aws_storage_architect,
    database_architect as aws_database_architect
)
from nodes.aws.validators import (
    validator_supervisor as aws_validator_supervisor,
    compute_validator as aws_compute_validator,
    network_validator as aws_network_validator,
    storage_validator as aws_storage_validator,
    database_validator as aws_database_validator
)

# Azure Nodes
from nodes.azure.architects import (
    azure_compute_architect,
    azure_network_architect,
    azure_storage_architect,
    azure_database_architect
)
from nodes.azure.validators import (
    azure_validator_supervisor,
    azure_compute_validator,
    azure_network_validator,
    azure_storage_validator,
    azure_database_validator
)

import logging

logger = logging.getLogger(__name__)


def create_graph_builder(llm_manager, tool_manager, provider: str = "AWS"):
    """
    Build the architecture generation graph builder (StateGraph).
    
    Args:
        llm_manager: Manager for LLM instances
        tool_manager: Manager for tools
        provider: "AWS", "Azure", or "All"
    
    Returns:
        StateGraph: The uncompiled graph builder
    """
    
    logger.info(f"Creating architecture graph builder for provider: {provider}")
    
    builder = StateGraph(ArchitectureState)
    
    # helper to normalize provider string
    p = provider.upper()
    include_aws = p in ["AWS", "ALL"]
    include_azure = p in ["AZURE", "ALL"]
    
    # ============ ADD NODES ============
    
    # 1. Supervisor (Shared)
    builder.add_node(
        "architect_supervisor",
        lambda state: architect_supervisor(state, llm_manager)
    )
    
    # 2. Architects (Provider Specific)
    if include_aws:
        builder.add_node("aws_compute_architect", lambda state: aws_compute_architect(state, llm_manager, tool_manager))
        builder.add_node("aws_network_architect", lambda state: aws_network_architect(state, llm_manager, tool_manager))
        builder.add_node("aws_storage_architect", lambda state: aws_storage_architect(state, llm_manager, tool_manager))
        builder.add_node("aws_database_architect", lambda state: aws_database_architect(state, llm_manager, tool_manager))
        
    if include_azure:
        builder.add_node("azure_compute_architect", lambda state: azure_compute_architect(state, llm_manager, tool_manager))
        builder.add_node("azure_network_architect", lambda state: azure_network_architect(state, llm_manager, tool_manager))
        builder.add_node("azure_storage_architect", lambda state: azure_storage_architect(state, llm_manager, tool_manager))
        builder.add_node("azure_database_architect", lambda state: azure_database_architect(state, llm_manager, tool_manager))

    # 3. Synthesizer (Shared)
    builder.add_node(
        "architect_synthesizer",
        lambda state: architect_synthesizer(state, llm_manager)
    )
    
    # 4. Validator Supervisor (Provider Specific or Shared?)
    # We have separate modules. We can add both if "ALL".
    if include_aws:
        builder.add_node(
            "aws_validator_supervisor",
            lambda state: aws_validator_supervisor(state, llm_manager)
        )
    
    if include_azure:
        builder.add_node(
            "azure_validator_supervisor",
            lambda state: azure_validator_supervisor(state, llm_manager)
        )
        
    # 5. Validators (Provider Specific)
    if include_aws:
        builder.add_node("aws_compute_validator", lambda state: aws_compute_validator(state, llm_manager, tool_manager))
        builder.add_node("aws_network_validator", lambda state: aws_network_validator(state, llm_manager, tool_manager))
        builder.add_node("aws_storage_validator", lambda state: aws_storage_validator(state, llm_manager, tool_manager))
        builder.add_node("aws_database_validator", lambda state: aws_database_validator(state, llm_manager, tool_manager))

    if include_azure:
        builder.add_node("azure_compute_validator", lambda state: azure_compute_validator(state, llm_manager, tool_manager))
        builder.add_node("azure_network_validator", lambda state: azure_network_validator(state, llm_manager, tool_manager))
        builder.add_node("azure_storage_validator", lambda state: azure_storage_validator(state, llm_manager, tool_manager))
        builder.add_node("azure_database_validator", lambda state: azure_database_validator(state, llm_manager, tool_manager))
    
    # 6. Validation Synthesizer (Shared)
    builder.add_node(
        "validation_synthesizer",
        lambda state: validation_synthesizer(state, llm_manager)
    )
    
    # 7. Final Generator (Shared)
    builder.add_node(
        "final_architecture_generator",
        lambda state: final_architecture_generator(state, llm_manager)
    )
    
    # ============ ADD EDGES ============
    
    # Start -> Supervisor
    builder.add_edge(START, "architect_supervisor")
    
    # Supervisor -> Architects
    if include_aws:
        builder.add_edge("architect_supervisor", "aws_compute_architect")
        builder.add_edge("architect_supervisor", "aws_network_architect")
        builder.add_edge("architect_supervisor", "aws_storage_architect")
        builder.add_edge("architect_supervisor", "aws_database_architect")
        
    if include_azure:
        builder.add_edge("architect_supervisor", "azure_compute_architect")
        builder.add_edge("architect_supervisor", "azure_network_architect")
        builder.add_edge("architect_supervisor", "azure_storage_architect")
        builder.add_edge("architect_supervisor", "azure_database_architect")
        
    # Architects -> Synthesizer
    if include_aws:
        builder.add_edge("aws_compute_architect", "architect_synthesizer")
        builder.add_edge("aws_network_architect", "architect_synthesizer")
        builder.add_edge("aws_storage_architect", "architect_synthesizer")
        builder.add_edge("aws_database_architect", "architect_synthesizer")

    if include_azure:
        builder.add_edge("azure_compute_architect", "architect_synthesizer")
        builder.add_edge("azure_network_architect", "architect_synthesizer")
        builder.add_edge("azure_storage_architect", "architect_synthesizer")
        builder.add_edge("azure_database_architect", "architect_synthesizer")
        
    # Synthesizer -> Validator Supervisors
    if include_aws:
        builder.add_edge("architect_synthesizer", "aws_validator_supervisor")
    if include_azure:
        builder.add_edge("architect_synthesizer", "azure_validator_supervisor")
        
    # Validator Supervisors -> Validators
    if include_aws:
        builder.add_edge("aws_validator_supervisor", "aws_compute_validator")
        builder.add_edge("aws_validator_supervisor", "aws_network_validator")
        builder.add_edge("aws_validator_supervisor", "aws_storage_validator")
        builder.add_edge("aws_validator_supervisor", "aws_database_validator")
        
    if include_azure:
        builder.add_edge("azure_validator_supervisor", "azure_compute_validator")
        builder.add_edge("azure_validator_supervisor", "azure_network_validator")
        builder.add_edge("azure_validator_supervisor", "azure_storage_validator")
        builder.add_edge("azure_validator_supervisor", "azure_database_validator")
        
    # Validators -> Validation Synthesizer
    if include_aws:
        builder.add_edge("aws_compute_validator", "validation_synthesizer")
        builder.add_edge("aws_network_validator", "validation_synthesizer")
        builder.add_edge("aws_storage_validator", "validation_synthesizer")
        builder.add_edge("aws_database_validator", "validation_synthesizer")

    if include_azure:
        builder.add_edge("azure_compute_validator", "validation_synthesizer")
        builder.add_edge("azure_network_validator", "validation_synthesizer")
        builder.add_edge("azure_storage_validator", "validation_synthesizer")
        builder.add_edge("azure_database_validator", "validation_synthesizer")
        
    # Iteration Loop
    builder.add_conditional_edges(
        "validation_synthesizer",
        iteration_condition,
        {
            "iterate": "architect_supervisor",
            "finish": "final_architecture_generator"
        }
    )
    
    builder.add_edge("final_architecture_generator", END)
    
    logger.info("Graph builder created successfully")
    return builder

# Backward compatibility alias
def build_architecture_graph(llm_manager, tool_manager):
    return create_graph_builder(llm_manager, tool_manager, provider="AWS")