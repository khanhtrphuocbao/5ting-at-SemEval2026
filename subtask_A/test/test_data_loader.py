import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data_loader import DataLoader


def test_data_loader():
    print("="*70)
    print("Testing Data Loader")
    print("="*70)
    
    config = Config()
    loader = DataLoader()
    
    # Test with clapnq dataset
    dataset = 'clapnq'
    query_type = 'questions'
    
    # Test corpus loading
    print(f"\n1. Testing corpus loading for {dataset}...")
    corpus_file = config.get_corpus_file(dataset)
    print(f"Corpus file: {corpus_file}")
    
    try:
        corpus = loader.load_corpus(corpus_file)
        print(f"   SUCCESS: Loaded {len(corpus)} documents")
        
        # Show sample document
        sample_id = list(corpus.keys())[0]
        sample_doc = corpus[sample_id]
        print(f"\nSample document:")
        print(f"- ID: {sample_doc.doc_id}")
        print(f"- Title: {sample_doc.title[:50]}...")
        print(f"- Text length: {len(sample_doc.text)} chars")
        print(f"- Text preview: {sample_doc.text[:100]}...")
        
        # Show stats
        stats = loader.get_corpus_stats(corpus)
        print(f"\nCorpus statistics:")
        for key, value in stats.items():
            print(f"- {key}: {value}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test queries loading
    print(f"\n2. Testing queries loading for {dataset} - {query_type}...")
    queries_file = config.get_queries_file(dataset, query_type)
    print(f"Queries file: {queries_file}")
    
    try:
        queries = loader.load_queries(queries_file)
        print(f"SUCCESS: Loaded {len(queries)} queries")
        
        # Show sample query
        sample_id = list(queries.keys())[0]
        sample_query = queries[sample_id]
        print(f"\nSample query:")
        print(f"- ID: {sample_query.query_id}")
        print(f"- Text: {sample_query.text[:200]}...")
        
        # Show stats
        stats = loader.get_queries_stats(queries)
        print(f"\nQuery statistics:")
        for key, value in stats.items():
            print(f"- {key}: {value}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Test qrels loading
    print(f"\n3. Testing qrels loading for {dataset}...")
    qrels_file = config.get_qrels_file(dataset)
    print(f"Qrels file: {qrels_file}")
    
    try:
        qrels = loader.load_qrels(qrels_file)
        print(f"SUCCESS: Loaded qrels for {len(qrels)} queries")
        
        # Show sample qrel
        sample_id = list(qrels.keys())[0]
        sample_relevant = qrels[sample_id]
        print(f"\nSample qrel:")
        print(f"- Query ID: {sample_id}")
        print(f"- Relevant docs: {len(sample_relevant)}")
        print(f"- Doc IDs: {sample_relevant[:3]}...")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "="*70)
    print("Data Loader Test Complete")
    print("="*70)


if __name__ == '__main__':
    test_data_loader()
