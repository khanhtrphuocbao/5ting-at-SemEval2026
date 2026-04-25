import os
from pathlib import Path
from dotenv import load_dotenv
from device_utils import get_device


class Config:
    def __init__(self, env_path: str = None):
        if env_path is None:
            project_root = Path(__file__).parent.parent
            env_path = project_root / '.env'
        
        load_dotenv(env_path)
        
        # Model Configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
        self.max_seq_length = int(os.getenv('MAX_SEQ_LENGTH', '8192'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        
        # Index Configuration
        self.index_path = os.getenv('INDEX_PATH', 'subtask_A/indexes')
        self.similarity_metric = os.getenv('SIMILARITY_METRIC', 'cosine')
        
        # Retrieval Configuration
        self.top_k = int(os.getenv('RETRIEVAL_K', '10'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '100'))
        
        # Data Paths
        self.corpus_path = os.getenv('CORPUS_PATH', 'corpora/passage_level')
        self.queries_path = os.getenv('QUERIES_PATH', 'human/retrieval_tasks')
        self.qrels_path = os.getenv('QRELS_PATH', 'human/retrieval_tasks')
        self.output_path = os.getenv('OUTPUT_PATH', 'subtask_A/results')
        
        # Datasets
        datasets_str = os.getenv('DOMAINS', 'clapnq,cloud,fiqa,govt')
        self.datasets = [d.strip() for d in datasets_str.split(',')]
        
        # Query Types
        query_types_str = os.getenv('QUERY_TYPES', 'questions,rewrite,lastturn')
        self.query_types = [q.strip() for q in query_types_str.split(',')]
        
        # Processing Configuration
        self.num_workers = int(os.getenv('NUM_WORKERS', '4'))
        self.device = get_device()
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
    def get_corpus_file(self, dataset: str) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / self.corpus_path / f"{dataset}.jsonl"
    
    def get_queries_file(self, dataset: str, query_type: str) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / self.queries_path / dataset / f"{dataset}_{query_type}.jsonl"
    
    def get_qrels_file(self, dataset: str) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / self.qrels_path / dataset / 'qrels' / 'dev.tsv'
    
    def get_index_path(self, dataset: str) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / self.index_path / dataset
    
    def get_output_path(self, dataset: str, query_type: str) -> Path:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / self.output_path / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{query_type}_results.tsv"
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"model={self.embedding_model},\n"
            f"device={self.device},\n"
            f"datasets={self.datasets},\n"
            f"top_k={self.top_k}\n"
            f")"
        )
