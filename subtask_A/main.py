"""
Retrieval Pipeline for Subtask A - Multi-Turn RAG Benchmark

Pipeline Flow:
1. Load Configuration
2. Load Corpus 
3. Build/Load Index
4. Load Queries
5. Retrieve Documents (top-K)
6. Rerank Documents (LLM-based)
7. Get Final Results (top-N)
8. Save Results
9. Evaluate
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data_loader import DataLoader, Query, Document
from embedding_generator import EmbeddingGenerator
from index_builder import IndexBuilder
from retrieval_engine import RetrievalEngine
from reranker import LLMReranker, RETRIEVAL_K, FINAL_K

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    log_path = Path(__file__).parent / 'retrieval_pipeline.log'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )
    
    return logging.getLogger(__name__)


# STEP 1: INITIALIZE COMPONENTS
def initialize_components(config: Config, logger: logging.Logger) -> dict:
    logger.info("Initializing pipeline components...")
    # Data loader
    data_loader = DataLoader(logger=logger)
    
    # Embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding_model,
        device=config.device,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        logger=logger
    )
    
    # Index builder
    index_builder = IndexBuilder(
        embedding_dim=embedding_generator.get_embedding_dim(),
        similarity_metric=config.similarity_metric,
        logger=logger
    )
    
    # Retrieval engine
    retrieval_engine = RetrievalEngine(
        embedding_generator=embedding_generator,
        index_builder=index_builder,
        data_loader=data_loader,
        logger=logger
    )
    
    # LLM Reranker
    reranker = LLMReranker(
        retrieval_k=RETRIEVAL_K,
        final_k=FINAL_K,
        logger=logger
    )
    
    logger.info("Components initialized")
    
    return {
        'data_loader': data_loader,
        'embedding_generator': embedding_generator,
        'index_builder': index_builder,
        'retrieval_engine': retrieval_engine,
        'reranker': reranker
    }


# STEP 2: LOAD CORPUS
def load_corpus(data_loader: DataLoader, corpus_file: Path, logger: logging.Logger) -> Dict[str, Document]:
    logger.info(f"Loading corpus from {corpus_file}")
    corpus = data_loader.load_corpus(corpus_file)
    stats = data_loader.get_corpus_stats(corpus)
    logger.info(f"Corpus stats: {stats}")
    return corpus


# STEP 3: BUILD/LOAD INDEX
def build_or_load_index(
    retrieval_engine: RetrievalEngine,
    corpus: Dict[str, Document],
    index_path: Path,
    rebuild: bool,
    logger: logging.Logger
):
    if rebuild or not index_path.exists():
        logger.info("Building new FAISS index...")
        retrieval_engine.build_index_from_corpus(corpus, index_path)
    else:
        logger.info(f"Loading existing index from {index_path}")
        retrieval_engine.load_index(index_path, corpus)



# STEP 4: LOAD QUERIES
def load_queries(data_loader: DataLoader, input_file: Path, logger: logging.Logger ) -> Tuple[Dict[str, Query], Dict[str, Dict]]:
    logger.info(f"Loading queries from {input_file}")
    
    # Load queries as Query objects
    queries = data_loader.load_queries_from_mtrageval(input_file)
    
    # Load full input data for output formatting
    input_data = data_loader.load_mtrageval_input(input_file)
    
    stats = data_loader.get_queries_stats(queries)
    logger.info(f"Query stats: {stats}")
    
    # Show sample
    if queries:
        sample_id = list(queries.keys())[0]
        sample_text = queries[sample_id].text[:100] + "..." if len(queries[sample_id].text) > 100 else queries[sample_id].text
        logger.info(f"Sample: {sample_text}")
    
    return queries, input_data


# STEP 5: RETRIEVE DOCUMENTS
def retrieve_documents(
    retrieval_engine: RetrievalEngine,
    queries: Dict[str, Query],
    top_k: int,
    logger: logging.Logger
) -> Dict[str, List[Tuple[str, float]]]:
    """Retrieve top-K documents for each query using embedding similarity."""
    logger.info(f"Retrieving top-{top_k} documents...")
    results = retrieval_engine.retrieve(queries, top_k=top_k)
    logger.info(f"Retrieved documents for {len(results)} queries")
    return results


# HELPER: MERGE RETRIEVAL RESULTS
def merge_retrieval_results(
    results1: Dict[str, List[Tuple[str, float]]],
    results2: Dict[str, List[Tuple[str, float]]],
    logger: logging.Logger
) -> Dict[str, List[Tuple[str, float]]]:
    """Merge two retrieval result sets, removing duplicates by doc_id."""
    merged_results = {}
    
    for query_id in results1.keys():
        if query_id not in results2:
            merged_results[query_id] = results1[query_id]
            continue
        
        # Combine both result lists
        combined = results1[query_id][:]
        existing_ids = {doc_id for doc_id, _ in results1[query_id]}
        
        # Add results from results2 that aren't duplicates
        for doc_id, score in results2[query_id]:
            if doc_id not in existing_ids:
                combined.append((doc_id, score))
                existing_ids.add(doc_id)
        
        merged_results[query_id] = combined
    
    return merged_results


# STEP 6: RERANK DOCUMENTS
def rerank_documents(
    reranker: LLMReranker,
    queries: Dict[str, Query],
    retrieval_results: Dict[str, List[Tuple[str, float]]],
    corpus: Dict[str, Document],
    logger: logging.Logger
) -> Dict[str, List[Tuple[str, float]]]:
    """Rerank documents using LLM-based reranking."""
    logger.info(f"Applying LLM-based reranking...")
    num_input_docs = len(retrieval_results[list(retrieval_results.keys())[0]]) if retrieval_results else 0
    logger.info(f"Input: {num_input_docs} docs → Output: {FINAL_K} docs per query")
    
    reranked_results = {}
    total = len(queries)
    
    for idx, (query_id, query_obj) in enumerate(queries.items(), 1):
        if query_id in retrieval_results:
            reranked = reranker.rerank_results(
                query=query_obj.text,
                retrieval_results=retrieval_results[query_id],
                corpus=corpus
            )
            reranked_results[query_id] = reranked
            
            if idx % 50 == 0 or idx == total:
                logger.info(f"   Progress: {idx}/{total} queries reranked")
    
    logger.info(f"Reranking complete: {len(reranked_results)} queries processed")
    return reranked_results


# STEP 7: SAVE RESULTS
def save_results(
    retrieval_engine: RetrievalEngine,
    results: Dict[str, List[Tuple[str, float]]],
    corpus: Dict[str, Document],
    input_data: Dict[str, Dict],
    output_file: Path,
    logger: logging.Logger
):
    """Save results in both TSV and JSONL formats."""
    logger.info(f"Saving results...")
    
    # Save TSV format
    retrieval_engine.save_results(results, output_file, format='tsv')
    logger.info(f"TSV saved: {output_file}")
    
    # Save JSONL format (for submission)
    jsonl_file = str(output_file).replace('.tsv', '_submission.jsonl')
    retrieval_engine.save_mtrageval_output(results, jsonl_file, corpus, input_data)
    logger.info(f"JSONL saved: {jsonl_file}")



# STEP 8: EVALUATE RESULTS
def evaluate_results(
    retrieval_engine: RetrievalEngine,
    data_loader: DataLoader,
    results: Dict[str, List[Tuple[str, float]]],
    qrels_file: Path,
    logger: logging.Logger
) -> Dict[str, float]:
    """Evaluate retrieval results against ground truth."""
    logger.info(f"Evaluating results...")
    
    qrels = data_loader.load_qrels(qrels_file)
    metrics = retrieval_engine.evaluate(results, qrels)
    retrieval_engine.print_evaluation_metrics(metrics)
    
    return metrics


# STEP 9: PRINT SUMMARY & EXPORT CSV
def print_summary(all_metrics: Dict, logger: logging.Logger):
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY OF ALL RESULTS")
    logger.info("=" * 70)
    
    for dataset, query_metrics in all_metrics.items():
        logger.info(f"\nDataset: {dataset}")
        logger.info("-" * 70)
        
        for query_type, metrics in query_metrics.items():
            logger.info(f"\nQuery Type: {query_type}")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name:15s}: {value:.4f}")
    
    logger.info("\n" + "=" * 70)


def export_to_csv(all_metrics: Dict, output_file: Path, logger: logging.Logger):
    csv_data = []
    
    for dataset, query_metrics in all_metrics.items():
        for query_type, metrics in query_metrics.items():
            for metric_name, value in metrics.items():
                csv_data.append({
                    'Dataset': dataset,
                    'Query_Type': query_type,
                    'Metric': metric_name,
                    'Value': round(value, 4)
                })
    
    df = pd.DataFrame(csv_data)
    df_pivot = df.pivot_table(
        index=['Dataset', 'Query_Type'],
        columns='Metric',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    # Reorder columns
    metric_cols = [col for col in df_pivot.columns if col not in ['Dataset', 'Query_Type']]
    desired_order = ['Dataset', 'Query_Type'] + sorted(metric_cols)
    df_pivot = df_pivot[desired_order]
    
    df_pivot.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"CSV summary saved to {output_file}")


# PIPELINE: Process Single Dataset + Query Type
def run_pipeline(config: Config, components: dict, dataset: str, query_type: str, rebuild_index: bool, logger: logging.Logger
) -> Dict[str, float]:
    """
    Run the complete retrieval + reranking pipeline.
    
    Flow:
        Query → Retrieve(20) → Rerank(LLM) → Top(10) → Save → Evaluate
    """
    logger.info("=" * 70)
    logger.info(f"PIPELINE: {dataset} - {query_type}")
    logger.info("=" * 70)
    
    # Get file paths
    corpus_file = config.get_corpus_file(dataset)
    qrels_file = config.get_qrels_file(dataset)
    index_path = config.get_index_path(dataset)
    output_file = config.get_output_path(dataset, query_type)
    input_file = Path('subtask_A') / 'input_data' / f'{dataset}_{query_type}_input.jsonl'
    
    # Extract components
    data_loader = components['data_loader']
    retrieval_engine = components['retrieval_engine']
    reranker = components['reranker']
    
    # STEP 2: Load corpus
    corpus = load_corpus(data_loader, corpus_file, logger)
    
    # STEP 3: Build/Load index
    build_or_load_index(retrieval_engine, corpus, index_path, rebuild_index, logger)
    
    # STEP 4: Load queries
    queries, input_data = load_queries(data_loader, input_file, logger)
    
    # STEP 5: Retrieve documents (top-20)
    # Special handling for lastturn: combine with rewrite results
    if query_type == 'lastturn':
        logger.info("*** ENHANCED RETRIEVAL: Combining lastturn + rewrite results ***")
        
        # Retrieve with lastturn queries
        initial_results_lastturn = retrieve_documents(retrieval_engine, queries, top_k=RETRIEVAL_K, logger=logger)
        
        # Load and retrieve with rewrite queries
        rewrite_input_file = Path('subtask_A') / 'input_data' / f'{dataset}_rewrite_input.jsonl'
        logger.info(f"Loading rewrite queries from {rewrite_input_file}")
        rewrite_queries, _ = load_queries(data_loader, rewrite_input_file, logger)
        
        logger.info(f"Retrieving top-{RETRIEVAL_K} documents for rewrite queries...")
        initial_results_rewrite = retrieve_documents(retrieval_engine, rewrite_queries, top_k=RETRIEVAL_K, logger=logger)
        
        # Merge results (lastturn + rewrite), remove duplicates
        logger.info("Merging lastturn and rewrite results (removing duplicates by doc_id)...")
        initial_results = merge_retrieval_results(initial_results_lastturn, initial_results_rewrite, logger)
        
        # Log statistics
        sample_query_id = list(initial_results.keys())[0]
        merged_count = len(initial_results[sample_query_id])
        logger.info(f"Merged result size per query: ~{merged_count} documents (from {RETRIEVAL_K}*2 with deduplication)")
    else:
        # Normal retrieval for rewrite and questions
        initial_results = retrieve_documents(retrieval_engine, queries, top_k=RETRIEVAL_K, logger=logger)
    
    # STEP 6: Rerank documents (merged results → top-10)
    final_results = rerank_documents(reranker, queries, initial_results, corpus, logger)
    
    # STEP 7: Save results
    save_results(retrieval_engine, final_results, corpus, input_data, output_file, logger)
    
    # STEP 8: Evaluate
    metrics = evaluate_results(retrieval_engine, data_loader, final_results, qrels_file, logger)
    
    logger.info(f"Pipeline completed for {dataset} - {query_type}")
    logger.info("=" * 70 + "\n")
    
    return metrics

# RUN ALL DATASETS
def run_all_datasets( config: Config, components: dict, rebuild_index: bool, logger: logging.Logger) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run pipeline for all datasets and query types."""
    logger.info("Starting retrieval pipeline for all datasets")
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Query types: {config.query_types}")
    
    all_metrics = {}
    
    for dataset in config.datasets:
        all_metrics[dataset] = {}
        for query_type in config.query_types:
            try:
                metrics = run_pipeline(config, components, dataset, query_type, rebuild_index, logger)
                all_metrics[dataset][query_type] = metrics
            except Exception as e:
                logger.error(f"Error processing {dataset} - {query_type}: {e}")
                import traceback
                traceback.print_exc()
    
    return all_metrics


# MAIN ENTRY POINT
def main():
    
    # STEP 1: Load Configuration
    config = Config()
    logger = setup_logger(config.log_level)
    
    logger.info("=" * 70)
    logger.info("MULTI-TURN RAG RETRIEVAL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"Embedding Model: {config.embedding_model}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Retrieval K: {RETRIEVAL_K}")
    logger.info(f"Final K (after rerank): {FINAL_K}")
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Query Types: {config.query_types}")
    logger.info("=" * 70)
    
    # STEP 2: Initialize Components
    components = initialize_components(config, logger)
    
    # STEP 3: Run Pipeline for All Datasets
    all_metrics = run_all_datasets(
        config=config,
        components=components,
        rebuild_index=False,  # Set True to rebuild indexes
        logger=logger
    )
    
    # STEP 4: Print Summary
    print_summary(all_metrics, logger)
    
    # STEP 5: Export Results to CSV
    csv_file = Path('subtask_A') / 'results_summary.csv'
    export_to_csv(all_metrics, csv_file, logger)
    
    logger.info("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
