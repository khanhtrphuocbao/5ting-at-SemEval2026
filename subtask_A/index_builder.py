import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import faiss

class IndexBuilder:
    def __init__(self, embedding_dim: int, similarity_metric: str = 'cosine', logger: logging.Logger = None):
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.logger = logger or logging.getLogger(__name__)
        self.index = None
        self.doc_ids = []
    
    def build_index(self, embeddings: np.ndarray, doc_ids: List[str]) -> faiss.Index:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings (num_docs, embedding_dim)
            doc_ids: List of document IDs corresponding to embeddings
            
        Returns:
            FAISS index
        """
        if len(embeddings) != len(doc_ids):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of document IDs ({len(doc_ids)})"
            )
        
        self.logger.info(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Create index based on similarity metric
        if self.similarity_metric == 'cosine':
            # For cosine similarity, use Inner Product with normalized vectors
            # Embeddings should already be normalized from the encoder
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors to index
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        
        # Store document IDs
        self.doc_ids = doc_ids
        
        self.logger.info(
            f"Index built successfully. Total vectors: {self.index.ntotal}, "
            f"Metric: {self.similarity_metric}"
        )
        
        return self.index
    
    def save_index(self, index_dir: Path):
        """
        Save the FAISS index and document IDs to disk.
        
        Args:
            index_dir: Directory to save the index files
        """
        if self.index is None:
            raise ValueError("No index to save. Build an index first.")
        
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = index_dir / 'faiss.index'
        faiss.write_index(self.index, str(index_file))
        self.logger.info(f"Saved FAISS index to {index_file}")
        
        # Save document IDs
        doc_ids_file = index_dir / 'doc_ids.pkl'
        with open(doc_ids_file, 'wb') as f:
            pickle.dump(self.doc_ids, f)
        self.logger.info(f"Saved document IDs to {doc_ids_file}")
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'similarity_metric': self.similarity_metric,
            'num_vectors': self.index.ntotal,
            'num_doc_ids': len(self.doc_ids)
        }
        metadata_file = index_dir / 'metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        self.logger.info(f"Saved metadata to {metadata_file}")
    
    def load_index(self, index_dir: Path):
        """
        Load a FAISS index and document IDs from disk.
        
        Args:
            index_dir: Directory containing the index files
        """
        index_dir = Path(index_dir)
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # Load FAISS index
        index_file = index_dir / 'faiss.index'
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        self.logger.info(f"Loaded FAISS index from {index_file}")
        
        # Load document IDs
        doc_ids_file = index_dir / 'doc_ids.pkl'
        if not doc_ids_file.exists():
            raise FileNotFoundError(f"Document IDs file not found: {doc_ids_file}")
        
        with open(doc_ids_file, 'rb') as f:
            self.doc_ids = pickle.load(f)
        self.logger.info(f"Loaded {len(self.doc_ids)} document IDs")
        
        # Load metadata
        metadata_file = index_dir / 'metadata.pkl'
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            self.logger.info(f"Loaded metadata: {metadata}")
    
    def search(self, query_embeddings: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embeddings: Query embeddings (num_queries, embedding_dim)
            top_k: Number of top results to return
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or build an index first.")
        
        query_embeddings_float32 = query_embeddings.astype('float32')
        distances, indices = self.index.search(query_embeddings_float32, top_k)
        
        return distances, indices
    
    def get_doc_ids_from_indices(self, indices: np.ndarray) -> List[List[str]]:
        """
        Convert index positions to document IDs.
        
        Args:
            indices: Array of index positions (num_queries, top_k)
            
        Returns:
            List of lists of document IDs
        """
        results = []
        for query_indices in indices:
            doc_id_list = [
                self.doc_ids[idx] if 0 <= idx < len(self.doc_ids) else None
                for idx in query_indices
            ]
            results.append(doc_id_list)
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the index builder."""
        num_vectors = self.index.ntotal if self.index is not None else 0
        return (
            f"IndexBuilder(\n"
            f"embedding_dim={self.embedding_dim},\n"
            f"similarity_metric={self.similarity_metric},\n"
            f"num_vectors={num_vectors}\n"
            f")"
        )
