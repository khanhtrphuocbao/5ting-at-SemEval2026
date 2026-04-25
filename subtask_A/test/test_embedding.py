import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from embedding_generator import EmbeddingGenerator


def test_embedding_generator():
    print("="*70)
    print("Testing Embedding Generator")
    print("="*70)
    
    config = Config()
    
    print(f"\nInitializing embedding generator...")
    print(f"Model: {config.embedding_model}")
    print(f"Device: {config.device}")
    
    try:
        generator = EmbeddingGenerator(
            model_name=config.embedding_model,
            device=config.device,
            max_seq_length=config.max_seq_length,
            batch_size=config.batch_size
        )
        
        print(f"SUCCESS: Model loaded")
        print(f"Embedding dimension: {generator.get_embedding_dim()}")
        print(generator)
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # Test with sample texts
    print(f"\n1. Testing document encoding...")
    sample_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ]
    
    try:
        embeddings = generator.encode_documents(sample_docs, show_progress=False)
        print(f"SUCCESS: Encoded {len(sample_docs)} documents")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"L2 norms: {norms}")
        print(f"Normalized: {np.allclose(norms, 1.0)}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test with sample queries
    print(f"\n2. Testing query encoding...")
    sample_queries = [
        "What is machine learning?",
        "Tell me about Python programming"
    ]
    
    try:
        query_embeddings = generator.encode_queries(sample_queries, show_progress=False)
        print(f"SUCCESS: Encoded {len(sample_queries)} queries")
        print(f"Embeddings shape: {query_embeddings.shape}")
        
        # Test similarity
        print(f"\n3. Testing similarity computation...")
        similarities = np.dot(query_embeddings, embeddings.T)
        print(f"Similarity matrix shape: {similarities.shape}")
        print(f"Similarities:")
        for i, query in enumerate(sample_queries):
            print(f"\nQuery: {query}")
            for j, doc in enumerate(sample_docs):
                print(f" Doc {j+1}: {similarities[i, j]:.4f} - {doc[:50]}...")
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "="*70)
    print("Embedding Generator Test Complete")
    print("="*70)


if __name__ == '__main__':
    test_embedding_generator()
