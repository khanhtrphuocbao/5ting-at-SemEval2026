import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

deployment_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def extract_last_user_question(conversation_text: str) -> str:
    """
    Extract the last user question from the conversation text.
    Format: |user|: question text\n|agent|: response\n...
    """
    pattern = r'\|user\|:\s*([^\n]*)'
    matches = re.findall(pattern, conversation_text)
    
    if matches:
        return matches[-1].strip()
    return ""

def create_rewrite_prompt(conversation_text: str, last_question: str) -> str:
    """
    Create a prompt to rewrite the last question based on conversation history.
    """
    prompt = f"""You are an expert Query Rewriter for a Retrieval-Augmented Generation system. 
        Your task is to rewrite the "Current User Query" into a **Standalone Query** that is fully descriptive and can be understood without any conversation history.

        **CRITICAL INSTRUCTIONS:**
        1. **Resolve Pronouns:** Replace words like "it", "he", "they", "that", "this" with the specific entity names from the History (User or Agent text).
        2. **Carry Over Predicates:** If the user asks "What about X?", apply the *action* or *question* from the previous turn to entity X.
        3. **Resolve Agent References:** If the user refers to an entity introduced by the Agent (e.g., "that last one"), identify the specific entity from the Agent's response.
        4. **Maintain Constraints:** Keep specific dates, seasons, or locations mentioned in previous turns if they limit the scope (e.g., "2017 season").
        5. **Handle Corrections:** If the user says "No, I meant X", rewrite the query to focus entirely on X, ignoring the mistake.

        **EXAMPLES:**
        History:
        User: Who is the CEO of Apple?
        Agent: Tim Cook.
        Query: What about Microsoft?
        Rewrite: Who is the CEO of Microsoft?

        History:
        User: Tell me about the 2017 Arizona Cardinals season.
        Agent: [Details about 2017 games...]
        Query: Did they play in London?
        Rewrite: Did the Arizona Cardinals play in London during the 2017 season?

        History:
        User: What films is Doctor Strange in?
        Agent: He appears in Doctor Strange, Thor: Ragnarok, and Avengers: Infinity War.
        Query: When was that last one released?
        Rewrite: When was the film Avengers: Infinity War released?

        History:
        User: Price of the Pixel 7.
        Query: No, I meant the Pixel 8.
        Rewrite: Price of the Pixel 8.

        **YOUR TASK:**
        CONVERSATION HISTORY:
        {conversation_text}

        ORIGINAL LAST QUESTION FROM USER:
        {last_question}

        Respond ONLY with the rewritten standalone query, nothing else."""

    return prompt

def rewrite_question_with_llm(conversation_text: str, last_question: str) -> str:
    """
    Use OpenAI to rewrite the last question based on conversation history.
    """
    prompt = create_rewrite_prompt(conversation_text, last_question)
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8120,
        )
        
        rewritten = response.choices[0].message.content.strip()
        return rewritten
    except Exception as e:
        print(f"Error: {e}")
        return last_question

def load_id_mapping(dataset: str) -> dict:
    """
    Load _id mapping from lastturn.jsonl file.
    """
    mapping = {}
    lastturn_file = f'human/retrieval_tasks_gen/{dataset}/{dataset}_lastturn.jsonl'
    
    try:
        with open(lastturn_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                _id = data['_id']
                text = data['text']
                parts = _id.split('<::>')
                prefix = parts[0]
                turn_num = int(parts[1]) if len(parts) > 1 else 1
                mapping[text] = {'prefix': prefix, 'turn': turn_num}
    except FileNotFoundError:
        print(f"Warning: {lastturn_file} not found")
    
    return mapping

def process_dataset(dataset: str):
    """
    Process one dataset: read fullconv.jsonl, rewrite questions, save results.
    """
    print(f"\n{'='*70}")
    print(f"Processing {dataset}")
    print(f"{'='*70}")
    
    fullconv_file = f'human/retrieval_tasks_gen/{dataset}/{dataset}_fullconv.jsonl'
    output_file = f'human/retrieval_tasks_gen/{dataset}/{dataset}_rewrite.jsonl'
    
    id_mapping = load_id_mapping(dataset)
    print(f"Loaded {len(id_mapping)} ID mappings from lastturn.jsonl")
    
    if not os.path.exists(fullconv_file):
        print(f"Error: {fullconv_file} not found")
        return
    
    rewrite_count = 0
    processed_count = 0
    unmapped_count = 0
    error_count = 0
    
    with open(fullconv_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line)
                conversation_text = data['text']
                
                last_user_question = extract_last_user_question(conversation_text)
                
                if not last_user_question:
                    print(f"Line {line_num}: Could not extract user question")
                    error_count += 1
                    continue
                
                _id_prefix = None
                turn_number = None
                
                for mapped_text, id_info in id_mapping.items():
                    if last_user_question in mapped_text:
                        _id_prefix = id_info['prefix']
                        turn_number = id_info['turn']
                        break
                
                if not _id_prefix:
                    unmapped_count += 1
                    print(f"Line {line_num}: No mapping for: {last_user_question[:50]}...")
                    continue
                
                print(f"[{line_num:3d}] Rewriting: {last_user_question[:60]}...", end=" ", flush=True)
                rewritten_text = rewrite_question_with_llm(conversation_text, last_user_question)
                print("[DONE]")
                
                output_data = {
                    "_id": f"{_id_prefix}<::>{turn_number}",
                    "text": f"|user|: {rewritten_text}",
                }
                
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                rewrite_count += 1
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                error_count += 1
    
    print(f"\n{dataset} Summary:")
    print(f"  OK Processed: {processed_count}")
    print(f"  OK Rewritten: {rewrite_count}")
    print(f"  FAIL Unmapped: {unmapped_count}")
    print(f"  FAIL Errors: {error_count}")
    print(f"  -> Output: {output_file}")

def main():
    """
    Main function to process all datasets.
    """
    datasets = ['clapnq', 'cloud', 'fiqa', 'govt']
    
    print(f"\n{'='*70}")
    print("Starting question rewriting with OpenAI LLM")
    print(f"Model: {deployment_name}")
    print(f"{'='*70}")
    
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    print(f"\n{'='*70}")
    print("OK All datasets processed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
