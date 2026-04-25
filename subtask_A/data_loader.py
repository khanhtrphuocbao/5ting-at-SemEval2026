import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    url: str = ""

@dataclass
class Query:
    query_id: str
    text: str

class DataLoader:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_corpus(self, corpus_file: Path) -> Dict[str, Document]:
        self.logger.info(f"Loading corpus from {corpus_file}")
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        corpus = {}
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        doc_id=data.get('_id', data.get('id', '')),
                        text=data.get('text', ''),
                        title=data.get('title', ''),
                        url=data.get('url', '')
                    )
                    corpus[doc.doc_id] = doc
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(corpus)} documents from corpus")
        return corpus
    
    def load_queries(self, queries_file: Path) -> Dict[str, Query]:
        self.logger.info(f"Loading queries from {queries_file}")
        
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    query = Query(
                        query_id=data.get('_id', ''),
                        text=data.get('text', '')
                    )
                    queries[query.query_id] = query
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(queries)} queries")
        return queries
    
    def load_qrels(self, qrels_file: Path) -> Dict[str, List[str]]:
        self.logger.info(f"Loading qrels from {qrels_file}")
        
        if not qrels_file.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_file}")
        
        qrels = {}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[1]
                    score = int(parts[2])
                    
                    if score > 0:  # Only consider relevant documents
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(doc_id)
        
        self.logger.info(f"Loaded qrels for {len(qrels)} queries")
        return qrels
    
    def get_corpus_stats(self, corpus: Dict[str, Document]) -> Dict[str, any]:
        total_docs = len(corpus)
        total_chars = sum(len(doc.text) for doc in corpus.values())
        avg_length = total_chars / total_docs if total_docs > 0 else 0
        
        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'avg_document_length': avg_length
        }
    
    def get_queries_stats(self, queries: Dict[str, Query]) -> Dict[str, any]:
        total_queries = len(queries)
        total_chars = sum(len(q.text) for q in queries.values())
        avg_length = total_chars / total_queries if total_queries > 0 else 0
        
        return {
            'total_queries': total_queries,
            'total_characters': total_chars,
            'avg_query_length': avg_length
        }
    
    def load_mtrageval_input(self, input_file: Path) -> Dict[str, Dict]:
        self.logger.info(f"Loading MTRAGEval input from {input_file}")
        
        if not input_file.exists():
            self.logger.warning(f"MTRAGEval input file not found: {input_file}")
            return {}
        
        input_data = {}
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    task_id = data.get('task_id', '')
                    input_data[task_id] = data
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(input_data)} tasks from MTRAGEval input")
        return input_data
    
    def load_queries_from_mtrageval(self, input_file: Path) -> Dict[str, Query]:
        self.logger.info(f"Loading queries from MTRAGEval input: {input_file}")
        
        if not input_file.exists():
            raise FileNotFoundError(f"MTRAGEval input file not found: {input_file}")
        
        queries = {}
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    task_id = data.get('task_id', '')
                    input_list = data.get('input', [])
                    
                    # Concatenate all user utterances to create query text
                    user_texts = []
                    for turn in input_list:
                        if turn.get('speaker') == 'user':
                            user_texts.append(turn.get('text', ''))
                    
                    query_text = '\n'.join(user_texts)
                    
                    query = Query(
                        query_id=task_id,
                        text=query_text
                    )
                    queries[task_id] = query
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        self.logger.info(f"Loaded {len(queries)} queries from MTRAGEval input")
        return queries
