# ============================================================================
# FILE: main.py
# PURPOSE: Entry point for the system
# ============================================================================

import logging
from typing import Generator, Dict, Any, Optional
import time

from langgraph.checkpoint.memory import MemorySaver
from core.types import ArchitectureState, create_initial_state
from core.tools import ToolManager, LLMManager
from graph.builder import create_graph_builder

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
            provider="AWS",
            thread_id="run-1"
        )
        print(result["final_architecture"]["document"])
    """
    
    def __init__(self):
        """Initialize the system with managers."""
        logger.info("Initializing Architecture Generation System...")
        
        self.tool_manager = ToolManager()
        self.llm_manager = LLMManager()
        # We don't pre-compile the graph anymore because it depends on the provider.
        # We'll compile it on demand or cache different versions.
        self.graphs = {} 
        
        logger.info("System initialized successfully")
    
    def get_graph(self, provider: str):
        """Get or create a compiled graph for the specified provider."""
        provider_key = provider.upper()
        if provider_key not in self.graphs:
            logger.info(f"Compiling graph for {provider_key}...")
            builder = create_graph_builder(self.llm_manager, self.tool_manager, provider=provider_key)
            checkpointer = MemorySaver()
            self.graphs[provider_key] = builder.compile(checkpointer=checkpointer)
        
        return self.graphs[provider_key]
    
    def run(
        self,
        user_problem: str,
        provider: str = "AWS",
        thread_id: Optional[str] = None,
        min_iterations: int = 2,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run architecture generation (blocking).
        
        Args:
            user_problem: Architecture problem to solve
            provider: Cloud provider ("AWS", "Azure", "All")
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
        logger.info(f"Starting Architecture Generation ({provider})")
        logger.info(f"Thread ID: {thread_id}")
        logger.info(f"Problem: {user_problem[:100]}...")
        logger.info(f"Iterations: min={min_iterations}, max={max_iterations}")
        logger.info(f"{'='*80}\n")
        
        # Create initial state with provider info
        initial_state = create_initial_state(
            user_problem, 
            min_iterations, 
            max_iterations,
            cloud_provider=provider
        )
        
        graph = self.get_graph(provider)
        result = graph.invoke(initial_state, config=config)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Architecture generation complete!")
        logger.info(f"Iterations completed: {result.get('iteration_count', 0)}")
        logger.info(f"Validation errors: {result.get('has_validation_errors', False)}")
        logger.info(f"Final architecture size: {len(result.get('architecture_summary', ''))} chars")
        logger.info(f"{'='*80}\n")
        
        return result
    
    def stream(
        self,
        user_problem: str,
        provider: str = "AWS",
        thread_id: Optional[str] = None,
        min_iterations: int = 2,
        max_iterations: int = 3
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream architecture generation (non-blocking, real-time updates).
        """
        
        if thread_id is None:
            thread_id = f"arch-stream-{int(time.time())}"
        
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = create_initial_state(
            user_problem, 
            min_iterations, 
            max_iterations,
            cloud_provider=provider
        )
        
        logger.info(f"Starting streaming generation (thread: {thread_id}, provider: {provider})")
        
        graph = self.get_graph(provider)
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            yield event


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    system = ArchitectureGenerationSystem()
    
    # Example 1: AWS Execution
    print("\n" + "="*80)
    print("EXAMPLE 1: AWS Execution")
    print("="*80)
    
    result_aws = system.run(
        user_problem="Design a scalable microservices platform.",
        provider="AWS",
        thread_id="example-aws",
        min_iterations=1,
        max_iterations=2
    )
    
    # Example 2: Azure Execution
    print("\n" + "="*80)
    print("EXAMPLE 2: Azure Execution")
    print("="*80)
    
    result_azure = system.run(
        user_problem="Design a scalable microservices platform.",
        provider="Azure",
        thread_id="example-azure",
        min_iterations=1,
        max_iterations=2
    )

