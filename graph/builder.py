# ============================================================================
# FILE: graph/builder.py
# PURPOSE: Construct the LangGraph workflow
# ============================================================================

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from core.types import ArchitectureState
from nodes.supervisor import architect_supervisor
from nodes.architects import (
    compute_architect, network_architect, storage_architect, database_architect
)
from nodes.synthesizer import architect_synthesizer
from nodes.validators import (
    validator_supervisor, compute_validator, network_validator,
    storage_validator, database_validator, validation_synthesizer
)
from nodes.iteration import iteration_condition, final_architecture_generator
import logging

logger = logging.getLogger(__name__)


def build_architecture_graph(llm_manager, tool_manager):
    """
    Build the complete architecture generation graph.
    
    WORKFLOW:
    1. Supervisor decomposes problem into tasks
    2. 4 Architects generate recommendations in parallel
    3. Synthesizer combines them
    4. Validator Supervisor creates validation tasks
    5. 4 Validators validate in parallel
    6. Validation Synthesizer decides: pass or fail?
    7. If fail: iterate back to supervisor
    8. If pass: generate final document
    """
    
    logger.info("Building architecture generation graph...")
    
    builder = StateGraph(ArchitectureState)
    
    # ============ ADD NODES ============
    
    # Supervisor
    builder.add_node(
        "architect_supervisor",
        lambda state: architect_supervisor(state, llm_manager)
    )
    
    # Architects (parallel)
    builder.add_node(
        "compute_architect",
        lambda state: compute_architect(state, llm_manager, tool_manager)
    )
    builder.add_node(
        "network_architect",
        lambda state: network_architect(state, llm_manager, tool_manager)
    )
    builder.add_node(
        "storage_architect",
        lambda state: storage_architect(state, llm_manager, tool_manager)
    )
    builder.add_node(
        "database_architect",
        lambda state: database_architect(state, llm_manager, tool_manager)
    )
    
    # Synthesizer
    builder.add_node(
        "architect_synthesizer",
        lambda state: architect_synthesizer(state, llm_manager)
    )
    
    # Validator Supervisor
    builder.add_node(
        "validator_supervisor",
        lambda state: validator_supervisor(state, llm_manager)
    )
    
    # Validators (parallel)
    builder.add_node(
        "compute_validator",
        lambda state: compute_validator(state, llm_manager, tool_manager)
    )
    builder.add_node(
        "network_validator",
        lambda state: network_validator(state, llm_manager, tool_manager)
    )
    builder.add_node(
        "storage_validator",
        lambda state: storage_validator(state, llm_manager, tool_manager)
    )
    builder.add_node(
        "database_validator",
        lambda state: database_validator(state, llm_manager, tool_manager)
    )
    
    # Validation Synthesizer
    builder.add_node(
        "validation_synthesizer",
        lambda state: validation_synthesizer(state, llm_manager)
    )
    
    # Final Generator
    builder.add_node(
        "final_architecture_generator",
        lambda state: final_architecture_generator(state, llm_manager)
    )
    
    # ============ ADD EDGES - GENERATION PHASE ============
    
    builder.add_edge(START, "architect_supervisor")
    
    # Parallel architects
    builder.add_edge("architect_supervisor", "compute_architect")
    builder.add_edge("architect_supervisor", "network_architect")
    builder.add_edge("architect_supervisor", "storage_architect")
    builder.add_edge("architect_supervisor", "database_architect")
    
    # Converge at synthesizer
    builder.add_edge("compute_architect", "architect_synthesizer")
    builder.add_edge("network_architect", "architect_synthesizer")
    builder.add_edge("storage_architect", "architect_synthesizer")
    builder.add_edge("database_architect", "architect_synthesizer")
    
    # ============ ADD EDGES - VALIDATION PHASE ============
    
    builder.add_edge("architect_synthesizer", "validator_supervisor")
    
    # Parallel validators
    builder.add_edge("validator_supervisor", "compute_validator")
    builder.add_edge("validator_supervisor", "network_validator")
    builder.add_edge("validator_supervisor", "storage_validator")
    builder.add_edge("validator_supervisor", "database_validator")
    
    # Converge at validation synthesizer
    builder.add_edge("compute_validator", "validation_synthesizer")
    builder.add_edge("network_validator", "validation_synthesizer")
    builder.add_edge("storage_validator", "validation_synthesizer")
    builder.add_edge("database_validator", "validation_synthesizer")
    
    # ============ ITERATION LOOP ============
    
    builder.add_conditional_edges(
        "validation_synthesizer",
        iteration_condition,
        {
            "iterate": "architect_supervisor",
            "finish": "final_architecture_generator"
        }
    )
    
    builder.add_edge("final_architecture_generator", END)
    
    # ============ COMPILE ============
    
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    logger.info("Graph built and compiled successfully")
    
    return graph