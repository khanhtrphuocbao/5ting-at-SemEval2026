import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from data_loader import DataLoader, Document, Query
from embedding_generator import EmbeddingGenerator
from index_builder import IndexBuilder
import json

class RetrievalEngine:
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        index_builder: IndexBuilder,
        data_loader: DataLoader,
        logger: logging.Logger = None
    ):
        self.embedding_generator = embedding_generator
        self.index_builder = index_builder
        self.data_loader = data_loader
        self.logger = logger or logging.getLogger(__name__)
        
        self.corpus = None
    
    def build_index_from_corpus(
        self,
        corpus: Dict[str, Document],
        index_save_path: Path = None
    ):
        """
        Build an index from a corpus of documents.
        
        Args:
            corpus: Dictionary mapping document IDs to Document objects
            index_save_path: Path to save the index (optional)
        """
        self.logger.info(f"Building index from {len(corpus)} documents...")
        
        # Store corpus for later use
        self.corpus = corpus
        
        # Extract document texts and IDs in consistent order
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[doc_id].text for doc_id in doc_ids]
        
        # Generate embeddings
        embeddings = self.embedding_generator.encode_documents(doc_texts)
        
        # Build index
        self.index_builder.build_index(embeddings, doc_ids)
        
        # Save index if path provided
        if index_save_path:
            self.index_builder.save_index(index_save_path)
    
    def load_index(self, index_path: Path, corpus: Dict[str, Document] = None):
        """
        Load a pre-built index from disk.
        
        Args:
            index_path: Path to the index directory
            corpus: Optional corpus dictionary for reference
        """
        self.logger.info(f"Loading index from {index_path}")
        self.index_builder.load_index(index_path)
        
        if corpus:
            self.corpus = corpus
    
    def retrieve(
        self,
        queries: Dict[str, Query],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve relevant documents for a set of queries.
        
        Args:
            queries: Dictionary mapping query IDs to Query objects
            top_k: Number of top documents to retrieve per query
            
        Returns:
            Dictionary mapping query IDs to lists of (doc_id, score) tuples
        """
        self.logger.info(f"Retrieving top-{top_k} documents for {len(queries)} queries...")
        
        # Extract query texts and IDs in consistent order
        query_ids = list(queries.keys())
        query_texts = [queries[qid].text for qid in query_ids]
        
        # Generate query embeddings
        query_embeddings = self.embedding_generator.encode_queries(query_texts)
        
        # Search index
        distances, indices = self.index_builder.search(query_embeddings, top_k)
        
        # Convert indices to document IDs
        doc_ids_list = self.index_builder.get_doc_ids_from_indices(indices)
        
        # Build results dictionary
        results = {}
        for i, query_id in enumerate(query_ids):
            results[query_id] = [
                (doc_id, float(score))
                for doc_id, score in zip(doc_ids_list[i], distances[i])
                if doc_id is not None
            ]
        
        self.logger.info(f"Retrieved results for {len(results)} queries")
        return results
    
    def save_results(self, results: Dict[str, List[Tuple[str, float]]], output_file: Path, format: str = 'tsv'):
        """
        Save retrieval results to a file.
        
        Args:
            results: Dictionary mapping query IDs to (doc_id, score) tuples
            output_file: Path to output file
            format: Output format ('tsv' or 'jsonl')
        """
        self.logger.info(f"Saving results to {output_file}")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'tsv':
            self._save_tsv(results, output_file)
        elif format == 'jsonl':
            self._save_jsonl(results, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_tsv(self, results: Dict[str, List[Tuple[str, float]]], output_file: Path):
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("query-id\tcorpus-id\tscore\n")
            
            # Write results
            for query_id, doc_scores in results.items():
                for doc_id, score in doc_scores:
                    f.write(f"{query_id}\t{doc_id}\t{score}\n")
        
        self.logger.info(f"Saved TSV results to {output_file}")
    
    def _save_jsonl(self, results: Dict[str, List[Tuple[str, float]]], output_file: Path):
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_id, doc_scores in results.items():
                result_obj = {
                    'query_id': query_id,
                    'results': [
                        {'doc_id': doc_id, 'score': score}
                        for doc_id, score in doc_scores
                    ]
                }
                f.write(json.dumps(result_obj) + '\n')
        
        self.logger.info(f"Saved JSONL results to {output_file}")
    
    def save_mtrageval_output(
        self,
        results: Dict[str, List[Tuple[str, float]]],
        output_file: Path,
        corpus: Dict[str, Document] = None,
        input_data: Dict[str, Dict] = None
    ):
        """
        Save results in MTRAGEval format for Subtask A submission.
        
        Format:
        {
            "conversation_id": "...",
            "task_id": "...",
            "Collection": "...",
            "input": [{"speaker": "user", "text": "..."}],
            "contexts": [
                {
                    "document_id": "...",
                    "text": "...",
                    "score": 27.759
                }
            ]
        }
        
        Args:
            results: Dictionary mapping query IDs to (doc_id, score) tuples
            output_file: Path to output file
            corpus: Corpus dictionary for document text
            input_data: MTRAGEval input data with task metadata
        """
        import json
        
        self.logger.info(f"Saving MTRAGEval format output to {output_file}")
        if input_data:
            self.logger.info(f"  Input data contains {len(input_data)} tasks")
            self.logger.info(f"  Results contain {len(results)} queries")
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        matched_count = 0
        unmatched_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_id, doc_scores in results.items():
                # Find matching input data
                task_data = None
                if input_data:
                    # Try exact match first
                    if query_id in input_data:
                        task_data = input_data[query_id]
                        matched_count += 1
                    else:
                        # Try partial match (query_id might be in task_id)
                        for task_id, data in input_data.items():
                            if query_id in task_id or task_id.endswith(query_id):
                                task_data = data
                                matched_count += 1
                                break
                        
                        if not task_data:
                            unmatched_count += 1
                
                # Build contexts list
                contexts = []
                for doc_id, score in doc_scores:
                    context = {
                        "document_id": doc_id,
                        "score": float(score)
                    }
                    
                    # Add document text if corpus available
                    if corpus and doc_id in corpus:
                        doc = corpus[doc_id]
                        context["text"] = doc.text
                        # title and source are optional for evaluation
                    
                    contexts.append(context)
                
                # Create output object
                output_obj = {}
                
                # Add metadata from input if available
                if task_data:
                    output_obj["conversation_id"] = task_data.get("conversation_id", "")
                    output_obj["task_id"] = task_data.get("task_id", query_id)
                    output_obj["Collection"] = task_data.get("Collection", "")
                    output_obj["input"] = task_data.get("input", [])
                else:
                    # Fallback if no input data available
                    output_obj["task_id"] = query_id
                
                output_obj["contexts"] = contexts
                
                f.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved MTRAGEval format results to {output_file}")
        if input_data:
            self.logger.info(f"Matched: {matched_count} queries")
            if unmatched_count > 0:
                self.logger.info(f"Unmatched: {unmatched_count} queries")
    
    def _calculate_dcg(self, relevances: List[int], k: int) -> float:
        dcg = 0.0
        for i, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(i + 1)
        return dcg
    
    def _calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Cut-off position
            
        Returns:
            nDCG@k score
        """
        # Create relevance list for retrieved documents
        relevances = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs[:k]]
        
        # Calculate DCG
        dcg = self._calculate_dcg(relevances, k)
        
        # Calculate ideal DCG (all relevant docs at top)
        ideal_relevances = [1] * min(len(relevant_docs), k) + [0] * max(0, k - len(relevant_docs))
        idcg = self._calculate_dcg(ideal_relevances, k)
        
        # Return normalized DCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def evaluate(self, results: Dict[str, List[Tuple[str, float]]], qrels: Dict[str, List[str]]) -> Dict[str, float]:
        self.logger.info("Evaluating retrieval results...")
        metrics = {
            'recall@1': 0.0,
            'recall@3': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'ndcg@1': 0.0,
            'ndcg@3': 0.0,
            'ndcg@5': 0.0,
            'ndcg@10': 0.0,
            'mrr': 0.0,
            'map': 0.0
        }
        
        num_queries = 0
        
        for query_id, relevant_docs in qrels.items():
            if query_id not in results:
                continue
            
            num_queries += 1
            retrieved_docs = [doc_id for doc_id, _ in results[query_id]]
            
            # Calculate recall and nDCG at different k values
            for k in [1, 3, 5, 10]:
                retrieved_at_k = set(retrieved_docs[:k])
                relevant_set = set(relevant_docs)
                recall = len(retrieved_at_k & relevant_set) / len(relevant_set) if relevant_set else 0
                metrics[f'recall@{k}'] += recall
                
                # Calculate nDCG@k
                ndcg = self._calculate_ndcg(retrieved_docs, relevant_docs, k)
                metrics[f'ndcg@{k}'] += ndcg
            
            # Calculate MRR (Mean Reciprocal Rank)
            for i, doc_id in enumerate(retrieved_docs[:100], 1):
                if doc_id in relevant_docs:
                    metrics['mrr'] += 1.0 / i
                    break
            
            # Calculate Average Precision
            num_relevant = 0
            precision_sum = 0.0
            for i, doc_id in enumerate(retrieved_docs[:100], 1):
                if doc_id in relevant_docs:
                    num_relevant += 1
                    precision_sum += num_relevant / i
            
            ap = precision_sum / len(relevant_docs) if relevant_docs else 0
            metrics['map'] += ap
        
        # Average metrics
        if num_queries > 0:
            for key in metrics:
                metrics[key] /= num_queries
        
        self.logger.info(f"Evaluation complete. Processed {num_queries} queries")
        return metrics
    
    def print_evaluation_metrics(self, metrics: Dict[str, float]):
        self.logger.info("\n" + "="*50)
        self.logger.info("EVALUATION METRICS")
        self.logger.info("="*50)
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name:15s}: {value:.4f}")
        self.logger.info("="*50 + "\n")
