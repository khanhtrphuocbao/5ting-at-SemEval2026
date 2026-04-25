import json
from pathlib import Path

# Collection mapping
COLLECTION_MAP = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'govt': 'mt-rag-govt-elser-512-100-20240611',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502'
}

def create_input_file(queries_file: Path, dataset: str, query_type: str, output_file: Path):
    if not queries_file.exists():
        print(f"Warning: Queries file not found: {queries_file}")
        return 0
    
    collection_name = COLLECTION_MAP.get(dataset, f'mt-rag-{dataset}-elser-512-100')
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    data_point_count = 0
    with open(queries_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            query_id = data.get('_id', '')
            query_text = data.get('text', '')
            
            # Create conversation_id from query_id (extract base part)
            conversation_id = query_id.split('_')[0] if '_' in query_id else query_id
            
            turn = conversation_id.split('<::>')[-1] if '<::>' in conversation_id else '1'
            conversation_id = conversation_id.split('<::>')[0]
            task_id = query_id #f"{conversation_id}<::>{turn}"
            
            # Parse multi-turn input from query text
            # Format: |user|: question 1\n|agent|: answer 1\n|user|: question 2
            input_list = []
            if '|user|:' in query_text or '|agent|:' in query_text:
                # Multi-turn conversation format
                parts = query_text.split('\n')
                for part in parts:
                    part = part.strip()
                    if part.startswith('|user|:'):
                        text = part.replace('|user|:', '').strip()
                        input_list.append({"speaker": "user", "text": text})
                    elif part.startswith('|agent|:'):
                        text = part.replace('|agent|:', '').strip()
                        input_list.append({"speaker": "agent", "text": text})
            else:
                # Single turn question
                input_list = [{"speaker": "user", "text": query_text}]
            
            # Create output object
            output_obj = {
                "conversation_id": conversation_id,
                "task_id": task_id,
                "Collection": collection_name,
                "input": input_list
            }
            
            f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            data_point_count += 1
    
    print(f"Created: {output_file.name:40} | Data points: {data_point_count:5}")
    return data_point_count


def main():
    """Create input files for all datasets and query types."""
    base_path = Path(__file__).parent.parent
    
    datasets = ['clapnq', 'cloud', 'fiqa', 'govt']
    query_types = ['questions', 'rewrite', 'lastturn']
    
    print("=" * 70)
    print("Creating MTRAGEval input files...")
    print("=" * 70)
    
    statistics = {}
    total_data_points = 0
    
    for dataset in datasets:
        statistics[dataset] = {}
        for query_type in query_types:
            # Input: human/retrieval_tasks/{dataset}/{dataset}_{query_type}.jsonl
            queries_file = base_path / 'human' / 'retrieval_tasks' / dataset / f'{dataset}_{query_type}.jsonl'
            
            # Output: subtask_A/input_data/{dataset}_{query_type}_input.jsonl
            output_file = base_path / 'subtask_A' / 'input_data' / f'{dataset}_{query_type}_input.jsonl'
            
            count = create_input_file(queries_file, dataset, query_type, output_file)
            statistics[dataset][query_type] = count
            total_data_points += count
    
    # Print summary statistics
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Questions':<15} {'Rewrite':<15} {'LastTurn':<15} {'Total':<10}")
    print("-" * 70)
    
    for dataset in datasets:
        questions = statistics[dataset].get('questions', 0)
        rewrite = statistics[dataset].get('rewrite', 0)
        lastturn = statistics[dataset].get('lastturn', 0)
        subtotal = questions + rewrite + lastturn
        print(f"{dataset:<15} {questions:<15} {rewrite:<15} {lastturn:<15} {subtotal:<10}")
    
    print("-" * 70)
    total_per_type = {
        'questions': sum(stat.get('questions', 0) for stat in statistics.values()),
        'rewrite': sum(stat.get('rewrite', 0) for stat in statistics.values()),
        'lastturn': sum(stat.get('lastturn', 0) for stat in statistics.values())
    }
    print(f"{'TOTAL':<15} {total_per_type['questions']:<15} {total_per_type['rewrite']:<15} {total_per_type['lastturn']:<15} {total_data_points:<10}")
    print("=" * 70)
    print(f"\nAll input files created successfully! ({total_data_points} total data points)")


if __name__ == '__main__':
    main()
