import logging
import numpy as np
from typing import List, Union

# Lazy import to handle PyTorch DLL issues on Windows
_sentence_transformers = None
_import_error = None


def _ensure_sentence_transformers_loaded():
    """Ensure SentenceTransformer is available."""
    global _sentence_transformers, _import_error
    
    if _sentence_transformers is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_transformers = SentenceTransformer
    except (ImportError, OSError) as e:
        _import_error = e

class EmbeddingGenerator:
    """
    Generates embeddings for documents and queries using SentenceTransformers.
    """
    
    def __init__(self, model_name: str, device: str = 'cuda', max_seq_length: int = 8192, 
                 batch_size: int = 32, logger: logging.Logger = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to use ('cuda' or 'cpu')
            max_seq_length: Maximum sequence length for the model
            batch_size: Batch size for encoding
            logger: Logger instance for logging
        """
        # Ensure SentenceTransformer is loaded
        _ensure_sentence_transformers_loaded()
        
        if _sentence_transformers is None:
            raise RuntimeError(
                f"Failed to import SentenceTransformer. This is usually caused by PyTorch DLL "
                f"loading issues on Windows. Error: {_import_error}\n")
        
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = _sentence_transformers(model_name, device=device)
        
        # Set max sequence length
        if hasattr(self.model, 'max_seq_length'):
            self.model.max_seq_length = max_seq_length
        
        self.logger.info(f"Model loaded on device: {device}")
        self.logger.info(f"Max sequence length: {max_seq_length}")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True, normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            NumPy array of embeddings with shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        self.logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        self.logger.info(f"Encoded {len(texts)} texts. Shape: {embeddings.shape}")
        return embeddings
    
    def encode_documents(self, documents: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode documents for indexing.
        
        Args:
            documents: List of document texts
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of document embeddings
        """
        self.logger.info(f"Encoding {len(documents)} documents for indexing...")
        return self.encode_texts(documents, show_progress=show_progress, normalize=True)
    
    def encode_queries(self, queries: List[str], show_progress: bool = True) -> np.ndarray:
        self.logger.info(f"Encoding {len(queries)} queries...")
        return self.encode_texts(queries, show_progress=show_progress, normalize=True)
    
    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        """String representation of the embedding generator."""
        return (
            f"EmbeddingGenerator(\n"
            f"model={self.model_name},\n"
            f"device={self.device},\n"
            f"max_seq_length={self.max_seq_length},\n"
            f"embedding_dim={self.get_embedding_dim()}\n"
            f")"
        )
