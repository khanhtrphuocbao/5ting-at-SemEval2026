import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import RetrievalPipeline
from config import Config


def test_single_dataset():
    print("="*70)
    print("Testing Retrieval Pipeline - Single Dataset")
    
    # Load configuration
    config = Config()
    print(f"\nConfiguration:")
    print(f"Model: {config.embedding_model}")
    print(f"Device: {config.device}")
    print(f"Top-K: {config.top_k}")
    print(f"Similarity: {config.similarity_metric}")
    
    # Create pipeline
    print(f"\nInitializing pipeline...")
    pipeline = RetrievalPipeline(config)
    
    # Test on a single dataset
    dataset = 'clapnq'  # Change this to test different datasets
    query_type = 'questions'  # Change this to test different query types
    
    print(f"\nRunning pipeline for: {dataset} - {query_type}")
    print(f"This will:")
    print(f"1. Load corpus from dataset")
    print(f"2. Build/load FAISS index")
    print(f"3. Load queries")
    print(f"4. Perform retrieval")
    print(f"5. Evaluate results")
    print()
    
    try:
        # Run pipeline (set rebuild_index=True to force rebuild)
        metrics = pipeline.run_for_dataset(
            dataset=dataset,
            query_type=query_type,
            rebuild_index=False  # Set to True to rebuild index
        )
        
        print("\n" + "="*70)
        print("Test Complete - Final Metrics:")
        print("="*70)
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s}: {value:.4f}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    test_single_dataset()
