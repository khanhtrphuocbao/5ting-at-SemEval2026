import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from index_builder import IndexBuilder
from retrieval_engine import RetrievalEngine


def example_1_quick_retrieval():
    """
    Example 1: Quick retrieval for a single dataset
    This is the simplest way to run retrieval.
    """
    print("="*70)
    print("Example 1: Quick Retrieval for Single Dataset")
    print("="*70)
    
    from main import RetrievalPipeline
    
    # Load configuration
    config = Config()
    
    # Create pipeline
    pipeline = RetrievalPipeline(config)
    
    # Run for a specific dataset and query type
    metrics = pipeline.run_for_dataset(
        dataset='clapnq',
        query_type='questions',
        rebuild_index=False  # Use existing index if available
    )
    
    print("\nMetrics:", metrics)


def example_2_build_custom_index():
    """
    Example 2: Build a custom index with specific settings
    Shows how to use individual components.
    """
    print("\n" + "="*70)
    print("Example 2: Building Custom Index")
    print("="*70)
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    data_loader = DataLoader()
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding_model,
        device=config.device,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size
    )
    index_builder = IndexBuilder(
        embedding_dim=embedding_generator.get_embedding_dim(),
        similarity_metric=config.similarity_metric
    )
    
    # Load corpus
    dataset = 'clapnq'
    corpus_file = config.get_corpus_file(dataset)
    print(f"\nLoading corpus from {corpus_file}...")
    corpus = data_loader.load_corpus(corpus_file)
    print(f"Loaded {len(corpus)} documents")
    
    # Extract texts and IDs
    doc_ids = list(corpus.keys())[:100]  # Use first 100 for demo
    doc_texts = [corpus[doc_id].text for doc_id in doc_ids]
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    embeddings = embedding_generator.encode_documents(doc_texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Build index
    print(f"\nBuilding index...")
    index = index_builder.build_index(embeddings, doc_ids)
    print(f"Index built with {index.ntotal} vectors")
    
    # Save index
    index_path = Path('subtask_A/indexes/custom_index')
    print(f"\nSaving index to {index_path}...")
    index_builder.save_index(index_path)
    print("Index saved successfully!")


def example_3_custom_retrieval():
    """
    Example 3: Custom retrieval with manual query processing
    Shows complete control over the retrieval process.
    """
    print("\n" + "="*70)
    print("Example 3: Custom Retrieval")
    print("="*70)
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    data_loader = DataLoader()
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding_model,
        device=config.device,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size
    )
    index_builder = IndexBuilder(
        embedding_dim=embedding_generator.get_embedding_dim(),
        similarity_metric=config.similarity_metric
    )
    
    # Load or build index
    dataset = 'clapnq'
    index_path = config.get_index_path(dataset)
    
    if index_path.exists():
        print(f"\nLoading existing index from {index_path}...")
        index_builder.load_index(index_path)
    else:
        print("\nIndex not found. Please run build_index.py first.")
        return
    
    # Custom queries
    custom_queries = {
        'query1': 'What is machine learning?',
        'query2': 'How does neural network training work?',
        'query3': 'Explain gradient descent algorithm'
    }
    
    print(f"\nProcessing {len(custom_queries)} custom queries...")
    
    # Encode queries
    query_ids = list(custom_queries.keys())
    query_texts = list(custom_queries.values())
    query_embeddings = embedding_generator.encode_queries(query_texts)
    
    # Search
    top_k = 5
    distances, indices = index_builder.search(query_embeddings, top_k)
    
    # Get document IDs
    doc_ids_list = index_builder.get_doc_ids_from_indices(indices)
    
    # Display results
    print(f"\nRetrieval Results (Top-{top_k}):")
    for i, (qid, qtext) in enumerate(zip(query_ids, query_texts)):
        print(f"\nQuery: {qtext}")
        print(f"Top {top_k} documents:")
        for rank, (doc_id, score) in enumerate(zip(doc_ids_list[i], distances[i]), 1):
            print(f"  {rank}. {doc_id}: {score:.4f}")


def example_4_evaluate_results():
    """
    Example 4: Evaluate retrieval results
    Shows how to compute metrics.
    """
    print("\n" + "="*70)
    print("Example 4: Evaluating Results")
    print("="*70)
    
    # Initialize configuration and components
    config = Config()
    data_loader = DataLoader()
    
    dataset = 'clapnq'
    query_type = 'questions'
    
    # Load qrels
    qrels_file = config.get_qrels_file(dataset)
    print(f"\nLoading ground truth from {qrels_file}...")
    qrels = data_loader.load_qrels(qrels_file)
    print(f"Loaded qrels for {len(qrels)} queries")
    
    # Mock results for demonstration
    # In practice, load these from retrieval output
    results = {}
    for query_id in list(qrels.keys())[:5]:  # First 5 queries
        relevant_docs = qrels[query_id]
        # Simulate retrieved docs (mix of relevant and irrelevant)
        retrieved = relevant_docs[:2] + ['dummy_doc_1', 'dummy_doc_2']
        results[query_id] = [(doc_id, 0.9 - i*0.1) for i, doc_id in enumerate(retrieved)]
    
    # Evaluate
    from retrieval_engine import RetrievalEngine
    
    retrieval_engine = RetrievalEngine(
        embedding_generator=None,
        index_builder=None,
        data_loader=data_loader
    )
    
    metrics = retrieval_engine.evaluate(results, qrels)
    
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:15s}: {value:.4f}")


def example_5_batch_processing():
    """
    Example 5: Batch processing multiple datasets
    Shows how to efficiently process all datasets.
    """
    print("\n" + "="*70)
    print("Example 5: Batch Processing All Datasets")
    print("="*70)
    
    from main import RetrievalPipeline
    
    # Load configuration
    config = Config()
    
    # Create pipeline
    pipeline = RetrievalPipeline(config)
    
    # Process all datasets and query types
    print("\nProcessing all datasets and query types...")
    print("This may take a while...\n")
    
    all_metrics = pipeline.run_all(rebuild_index=False)
    
    # Display summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    
    for dataset, query_metrics in all_metrics.items():
        print(f"\n{dataset}:")
        for query_type, metrics in query_metrics.items():
            print(f"  {query_type}:")
            print(f"    Recall@10: {metrics.get('recall@10', 0):.4f}")
            print(f"    MRR: {metrics.get('mrr', 0):.4f}")


def main():
    """
    Run selected examples.
    Uncomment the examples you want to run.
    """
    print("Subtask A - Usage Examples\n")
    
    # Choose which examples to run (uncomment to enable)
    
    # Example 1: Quick retrieval (recommended to start)
    # example_1_quick_retrieval()
    
    # Example 2: Build custom index
    # example_2_build_custom_index()
    
    # Example 3: Custom retrieval
    # example_3_custom_retrieval()
    
    # Example 4: Evaluate results
    # example_4_evaluate_results()
    
    # Example 5: Batch processing (full pipeline)
    # example_5_batch_processing()
    
    print("\nTo run examples, uncomment them in the main() function.")
    print("Recommended order:")
    print("1. example_1_quick_retrieval() - Simple single dataset run")
    print("2. example_3_custom_retrieval() - Custom queries")
    print("3. example_5_batch_processing() - Full pipeline")

if __name__ == '__main__':
    main()
