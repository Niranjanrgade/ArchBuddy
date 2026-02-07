# ============================================================================
# FILE: main.py
# PURPOSE: Entry point for the system
# ============================================================================

import logging
from typing import Generator, Dict, Any, Optional
import time

from core.types import ArchitectureState, create_initial_state
from core.tools import ToolManager, LLMManager
from graph.builder import build_architecture_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArchitectureGenerationSystem:
    """
    Main system for generating cloud architectures.
    
    USAGE:
        system = ArchitectureGenerationSystem()
        result = system.run(
            "Build a containerized microservices platform on AWS",
            thread_id="run-1"
        )
        print(result["final_architecture"]["document"])
    """
    
    def __init__(self):
        """Initialize the system with managers and graph."""
        logger.info("Initializing Architecture Generation System...")
        
        self.tool_manager = ToolManager()
        self.llm_manager = LLMManager()
        self.graph = build_architecture_graph(self.llm_manager, self.tool_manager)
        
        logger.info("System initialized successfully")
    
    def run(
        self,
        user_problem: str,
        thread_id: Optional[str] = None,
        min_iterations: int = 2,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run architecture generation (blocking).
        
        Args:
            user_problem: Architecture problem to solve
            thread_id: Optional thread ID for checkpointing
            min_iterations: Minimum refinement loops
            max_iterations: Maximum refinement loops
        
        Returns:
            Final state with architecture
        """
        
        if thread_id is None:
            thread_id = f"arch-{int(time.time())}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Architecture Generation")
        logger.info(f"Thread ID: {thread_id}")
        logger.info(f"Problem: {user_problem[:100]}...")
        logger.info(f"Iterations: min={min_iterations}, max={max_iterations}")
        logger.info(f"{'='*80}\n")
        
        initial_state = create_initial_state(user_problem, min_iterations, max_iterations)
        result = self.graph.invoke(initial_state, config=config)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Architecture generation complete!")
        logger.info(f"Iterations completed: {result['iteration_count']}")
        logger.info(f"Validation errors: {result['has_validation_errors']}")
        logger.info(f"Final architecture size: {len(result.get('architecture_summary', ''))} chars")
        logger.info(f"{'='*80}\n")
        
        return result
    
    def stream(
        self,
        user_problem: str,
        thread_id: Optional[str] = None,
        min_iterations: int = 2,
        max_iterations: int = 3
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream architecture generation (non-blocking, real-time updates).
        
        Yields state updates as they occur.
        """
        
        if thread_id is None:
            thread_id = f"arch-stream-{int(time.time())}"
        
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = create_initial_state(user_problem, min_iterations, max_iterations)
        
        logger.info(f"Starting streaming generation (thread: {thread_id})")
        
        for event in self.graph.stream(initial_state, config=config, stream_mode="updates"):
            yield event


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize system (once, reuse for multiple runs)
    system = ArchitectureGenerationSystem()
    
    # Example 1: Standard execution
    print("\n" + "="*80)
    print("EXAMPLE 1: Standard Execution (Blocking)")
    print("="*80)
    
    result = system.run(
        user_problem="""
Design a scalable microservices platform on AWS that:
- Handles 10,000 concurrent users
- Processes 1 million API requests/day
- Requires high availability (99.99% uptime)
- Must be cost-optimized
- Needs strong security and compliance
        """.strip(),
        thread_id="example-1",
        min_iterations=1,
        max_iterations=2
    )
    
    if result.get('final_architecture'):
        print(f"\n✓ Final Architecture ({len(result['final_architecture'].get('document', ''))} chars):")
        doc = result['final_architecture'].get('document', '')
        print(doc[:500] + ("..." if len(doc) > 500 else ""))
    else:
        print("\n✗ No final architecture generated")
    
    # Example 2: Streaming execution
    print("\n" + "="*80)
    print("EXAMPLE 2: Streaming Execution (Real-time)")
    print("="*80)
    
    event_count = 0
    nodes_executed = []
    
    for event in system.stream(
        user_problem="Build a real-time analytics platform on AWS with data processing pipelines",
        thread_id="example-2",
        min_iterations=1,
        max_iterations=2
    ):
        event_count += 1
        for node_name in event.keys():
            if node_name not in nodes_executed:
                nodes_executed.append(node_name)
                print(f"  [{event_count}] ✓ {node_name}")
    
    print(f"\nTotal events: {event_count}")
    print(f"Nodes executed: {len(nodes_executed)}")
    print(f"Nodes: {', '.join(nodes_executed)}")
