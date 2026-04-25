import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pydantic import BaseModel
from litellm import completion
from tenacity import retry, wait_exponential


# Configuration
RETRIEVAL_K = 20  # Initial retrieval count
FINAL_K = 10      # Final count after reranking
MODEL = "gpt-4o-mini"  # LLM model for reranking


class RankOrder(BaseModel):
    order: List[int]

@dataclass
class DocumentChunk:
    doc_id: str
    content: str
    score: float
    metadata: Dict = None

class LLMReranker:
    def __init__(
        self,
        model: str = MODEL,
        retrieval_k: int = RETRIEVAL_K,
        final_k: int = FINAL_K,
        logger: logging.Logger = None
    ):
        self.model = model
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.logger = logger or logging.getLogger(__name__)
        
        # Configure retry with exponential backoff
        self.wait_strategy = wait_exponential(multiplier=1, min=2, max=10)
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10))
    def rerank_chunks(self, question: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Rerank document chunks using LLM.
        
        Args:
            question: User's question
            chunks: List of document chunks to rerank
            
        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []
        

        system_prompt = """
        You are a document re-ranker.
        You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
        The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
        You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
        Reply ONLY with a JSON object containing "order": [list of chunk IDs from most to least relevant].
        Include ALL chunk IDs in your ranking.
        """
        
        user_prompt = f"The user has asked the following question: {question}\n\n"
        user_prompt += "Here are the chunks:\n\n"
        
        for index, chunk in enumerate(chunks, 1):
            user_prompt += f"CHUNK ID {index}:\n{chunk.content}\n\n"
            self.logger.info(f"_"*60)
            self.logger.info(f"CHUNK ID {index}: \n{chunk.content[:200]}")

            self.logger.info(f"_"*60)
        user_prompt += "Reply with JSON: {\"order\": [ranked chunk IDs]}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = completion(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            reply = response.choices[0].message.content
            rank_order = RankOrder.model_validate_json(reply)
            
            # Reorder chunks based on LLM ranking
            reranked = [chunks[i - 1] for i in rank_order.order if 1 <= i <= len(chunks)]
            
            self.logger.info(f"Reranked {len(chunks)} chunks. New order: {rank_order.order}...")
            return reranked
            
        except Exception as e:
            self.logger.warning(f"LLM reranking failed: {e}. Returning original order.")
            return chunks
    
    def merge_chunks(self,
        chunks1: List[DocumentChunk],
        chunks2: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Merge two lists of chunks, avoiding duplicates.
        
        Args:
            chunks1: First list of chunks
            chunks2: Second list of chunks
            
        Returns:
            Merged list without duplicates
        """
        merged = chunks1[:]
        existing_ids = {chunk.doc_id for chunk in chunks1}
        
        for chunk in chunks2:
            if chunk.doc_id not in existing_ids:
                merged.append(chunk)
                existing_ids.add(chunk.doc_id)
        
        return merged
    
    def rerank_results(
        self,
        query: str,
        retrieval_results: List[Tuple[str, float]],
        corpus: Dict[str, any]
    ) -> List[Tuple[str, float]]:
        """
        Main reranking pipeline for retrieval results.
        
        Pipeline:
        1. Retrieve top-K documents (already done, passed as retrieval_results)
        2. Rewrite query
        3. Could retrieve again with rewritten query (optional, skipped here)
        4. Rerank using LLM
        5. Return top-N after reranking
        
        Args:
            query: Original search query
            retrieval_results: List of (doc_id, score) tuples from initial retrieval
            corpus: Dictionary mapping doc_id to Document objects
            
        Returns:
            Reranked list of (doc_id, score) tuples, limited to final_k
        """
        if not retrieval_results:
            return []
        
        # Convert retrieval results to DocumentChunk objects
        chunks = []
        for doc_id, score in retrieval_results:
            if doc_id in corpus:
                doc = corpus[doc_id]
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    content=doc.text,
                    score=score,
                    metadata={'title': doc.title, 'url': doc.url}
                ))
        
        # Validate chunks from retrieval results
        if not chunks:
            self.logger.warning("No valid chunks found from retrieval results!")
            return []
        
        self.logger.info(f"Converted {len(chunks)} retrieval results to chunks for reranking")
        self.logger.info(f"Query: {query[:100]}...")
        self.logger.info(f"Chunk IDs: {[c.doc_id for c in chunks[:5]]}...")
        
        # Rerank using LLM - these are the actual chunks from retrieval
        reranked_chunks = self.rerank_chunks(query, chunks)
        
        # Convert back to (doc_id, score) format
        # Keep original scores but reorder by LLM ranking
        reranked_results = [
            (chunk.doc_id, chunk.score)
            for chunk in reranked_chunks[:self.final_k]
        ]
        
        self.logger.info(f"Reranking complete. Returning top {len(reranked_results)} documents.")
        return reranked_results
