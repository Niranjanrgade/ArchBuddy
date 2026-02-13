import logging
import sys
from main import ArchitectureGenerationSystem

# Configure logging to see output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify")

def test_providers():
    print("Initializing system...")
    try:
        system = ArchitectureGenerationSystem()
        
        # Test 1: AWS Graph Compilation
        print("\n--- Testing AWS Graph Compilation ---")
        graph_aws = system.get_graph("AWS")
        print("AWS Graph compiled successfully.")
        
        # Test 2: Azure Graph Compilation
        print("\n--- Testing Azure Graph Compilation ---")
        graph_azure = system.get_graph("Azure")
        print("Azure Graph compiled successfully.")
        
        # Test 3: All Graph Compilation
        print("\n--- Testing 'All' Graph Compilation ---")
        graph_all = system.get_graph("All")
        print("'All' Graph compiled successfully.")
        
        print("\nAll compilation tests passed!")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_providers()
