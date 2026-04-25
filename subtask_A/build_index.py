import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from index_builder import IndexBuilder
from retrieval_engine import RetrievalEngine


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def build_index_for_dataset(config: Config, dataset: str, logger: logging.Logger):
    """
    Build index for a specific dataset.
    
    Args:
        config: Configuration object
        dataset: Dataset name
        logger: Logger instance
    """
    logger.info(f"Building index for dataset: {dataset}")
    
    # Initialize components
    data_loader = DataLoader(logger=logger)
    embedding_generator = EmbeddingGenerator(
        model_name=config.embedding_model,
        device=config.device,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        logger=logger
    )
    index_builder = IndexBuilder(
        embedding_dim=embedding_generator.get_embedding_dim(),
        similarity_metric=config.similarity_metric,
        logger=logger
    )
    retrieval_engine = RetrievalEngine(
        embedding_generator=embedding_generator,
        index_builder=index_builder,
        data_loader=data_loader,
        logger=logger
    )
    
    # Get paths
    corpus_file = config.get_corpus_file(dataset)
    index_path = config.get_index_path(dataset)
    
    # Load corpus
    logger.info(f"Loading corpus from {corpus_file}")
    corpus = data_loader.load_corpus(corpus_file)
    
    # Build and save index
    logger.info("Building index...")
    retrieval_engine.build_index_from_corpus(corpus, index_path)
    
    logger.info(f"Index built and saved to {index_path}")


def main():
    """Main entry point for building indexes."""
    logger = setup_logger()
    config = Config()
    
    logger.info("Starting index building process")
    logger.info(f"Datasets to process: {config.datasets}")
    
    for dataset in config.datasets:
        try:
            build_index_for_dataset(config, dataset, logger)
        except Exception as e:
            logger.error(f"Error building index for {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Index building complete")


if __name__ == '__main__':
    main()
