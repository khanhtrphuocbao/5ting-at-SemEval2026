import os
import json
import argparse
from openai import OpenAI

def build_prompt(history: str, user_question: str, retrieved_chunks: str) -> str:
    """
    Builds the role-separated RAG prompt for grounded answer generation.

    Role architecture:
      - Conversation History : pragmatic context only (intent, coreference resolution)
      - Retrieved Passages   : sole source of factual information
      - Last-Turn Question   : the specific question to answer
    """
    history_block = history.strip() if history.strip() else "[No prior conversation history]"
    passages_block = retrieved_chunks.strip() if retrieved_chunks.strip() else "[No retrieved passages available]"

    prompt = f"""You are a precise and faithful question-answering assistant operating within a \
Retrieval-Augmented Generation (RAG) pipeline. Your sole task is to answer the user's \
final question using ONLY the factual information present in the Retrieved Passages provided below.

=== ROLE DEFINITIONS — READ CAREFULLY BEFORE ANSWERING ===

[ROLE 1 — CONVERSATION HISTORY: For context resolution ONLY]
The conversation history is provided exclusively to help you:
  • Understand what topic the conversation concerns.
  • Resolve coreferences and pronouns (e.g., "it", "that option", "the previous method").
  • Identify what has already been discussed or clarified in prior turns.

CRITICAL: The conversation history is NOT a source of factual information.
Do NOT extract, infer, or cite any facts from it. Treat it as pragmatic context only.

[ROLE 2 — RETRIEVED PASSAGES: The SOLE source of facts]
The retrieved passages below represent the only authoritative factual basis for your answer.
Every factual claim in your response MUST be directly supported by at least one of these passages.
If the passages do not contain sufficient information to answer the question, you must state that explicitly.

[ROLE 3 — LAST-TURN QUESTION: The specific question to answer]
Answer this question directly, concisely, and faithfully, grounded strictly in the retrieved passages.

=== CONVERSATION HISTORY (intent and reference resolution only — NOT a factual source) ===
{history_block}

=== RETRIEVED PASSAGES (the sole factual basis for your answer) ===
{passages_block}

=== LAST-TURN QUESTION ===
{user_question}

=== ANSWER GENERATION GUIDELINES ===
1. Ground every factual claim EXCLUSIVELY in the Retrieved Passages. Do not use parametric world knowledge.
2. If the passages are insufficient to answer the question, respond:
   "Based on the available retrieved information, I cannot provide a complete answer to this question."
3. Do NOT speculate, hallucinate, or introduce any information absent from the passages.
4. Use the Conversation History solely to resolve coreferences and clarify the intent of the question.
5. Synthesize information across passages coherently rather than quoting them verbatim.
6. Be concise and direct. Avoid unnecessary preamble or meta-commentary about the answer.
7. If the question is underspecified or ambiguous, acknowledge the ambiguity briefly before answering.

ANSWER:"""
    return prompt


def generate_answers(histories_path: str, retrieval_path: str, output_path: str, test: bool = True):
    """
    Generate LLM answers and save to JSONL file
    """
    client = OpenAI()
    
    # Load retrieval_map
    retrieval_map = {}
    if retrieval_path and os.path.exists(retrieval_path):
        try:
            with open(retrieval_path, 'r', encoding='utf-8') as rf:
                for rline in rf:
                    rline = rline.strip()
                    if not rline:
                        continue
                    try:
                        robj = json.loads(rline)
                    except json.JSONDecodeError:
                        continue
                    conv_id = robj.get('conversation_id') or robj.get('task_id')
                    contexts = robj.get('contexts', [])
                    if isinstance(contexts, list) and contexts:
                        chunks = '\n---\n'.join([
                            (c.get('text', '') if isinstance(c, dict) else str(c)).strip() 
                            for c in contexts
                        ])
                    else:
                        chunks = ''
                    if conv_id:
                        retrieval_map[conv_id] = chunks
        except Exception as e:
            print(f"Error loading retrieval file: {e}")
            return
    
    # Open output file for writing
    processed = 0
    with open(output_path, 'w', encoding='utf-8') as outf:
        with open(histories_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Lấy input
                inputs = obj.get('input', [])
                if not isinstance(inputs, list) or len(inputs) == 0:
                    continue
                
                # Tách lịch sử
                if len(inputs) > 1:
                    histories = inputs[:-1]
                    last_turn = inputs[-1]
                else:
                    histories = []
                    last_turn = inputs[0] if inputs else {}
                
                # Format histories
                if histories:
                    histories_str = "\n".join([
                        f"{h.get('speaker', 'unknown').upper()}: {h.get('text', '')}"
                        for h in histories
                    ])
                else:
                    histories_str = ""
                
                # Last question
                lastturn_text = last_turn.get('text', '') if isinstance(last_turn, dict) else str(last_turn)
                
                # Get chunks
                conv_id = obj.get('conversation_id') or obj.get('task_id')
                chunks_text = retrieval_map.get(conv_id, '')
                
                # Build prompt
                prompt = build_prompt(histories_str, lastturn_text, chunks_text)
                
                print(f"Processing {processed+1}... ID: {conv_id}", end=" ", flush=True)
                
                # Call LLM via Responses API
                try:
                    response = client.responses.create(
                        model="gpt-4-mini",
                        input=prompt,
                        max_tokens=1000,
                    )
                    try:
                        answer = response.output[0].content[0].text
                    except Exception:
                        answer = getattr(response, 'output_text', str(response))
                    print("✓ OK")
                    
                    # Save to output
                    result = {
                        "conversation_id": conv_id,
                        "task_id": obj.get('task_id'),
                        "collection": obj.get('Collection'),
                        "user_question": lastturn_text,
                        "conversation_history": histories_str,
                        "retrieved_chunks": chunks_text,
                        "full_prompt": prompt,
                        "llm_answer": answer
                    }
                    outf.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    result = {
                        "conversation_id": conv_id,
                        "task_id": obj.get('task_id'),
                        "error": str(e)
                    }
                    outf.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                processed += 1
                if test:
                    print(f"\n[TEST MODE] Processed {processed} record. Stopping.")
                    break
    
    if not test:
        print(f"\n[FULL MODE] Total records processed: {processed}")
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate LLM answers and save to JSONL')
    parser.add_argument('-histories', '--histories_file', 
                       default=os.path.join('subtask_A_testphase', 'data_test_phase', 'rag_taskAC_fullconversation.jsonl'))
    parser.add_argument('-retrieval', '--retrieval_file', 
                       default=os.path.join('subtask_C', 'A_top5', 'retrieval_top5.jsonl'))
    parser.add_argument('-o', '--output', 
                       default=os.path.join('subtask_C', 'A_top5', 'llm_answers.jsonl'))
    parser.add_argument('--test', action='store_true', help='Test mode: process only 1 record')
    args = parser.parse_args()
    
    generate_answers(args.histories_file, args.retrieval_file, args.output, test=args.test)


if __name__ == '__main__':
    main()
