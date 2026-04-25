import json
from pathlib import Path

def test_output_format(output_file):
    output_file = Path(output_file)
    if not output_file.exists():
        print(f"✗ Output file not found: {output_file}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Testing output format: {output_file.name}")

    required_fields = {'conversation_id', 'task_id', 'Collection', 'input', 'contexts'}
    context_fields = {'document_id', 'score'}
    
    total_lines = 0
    valid_lines = 0
    sample_lines = []
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                obj = json.loads(line.strip())
                
                # Check required fields
                missing_fields = required_fields - set(obj.keys())
                if missing_fields:
                    print(f"Line {line_num}: Missing fields: {missing_fields}")
                    continue
                
                # Check contexts structure
                if not isinstance(obj.get('contexts'), list):
                    print(f"  Line {line_num}: 'contexts' is not a list")
                    continue
                
                for ctx_idx, ctx in enumerate(obj['contexts']):
                    missing_ctx_fields = context_fields - set(ctx.keys())
                    if missing_ctx_fields:
                        print(f"Line {line_num}, Context {ctx_idx}: Missing fields: {missing_ctx_fields}")
                        continue
                    
                    # Verify score is numeric
                    if not isinstance(ctx['score'], (int, float)):
                        print(f"Line {line_num}, Context {ctx_idx}: 'score' is not numeric")
                        continue
                
                valid_lines += 1
                
                # Store first few samples for display
                if line_num <= 3:
                    sample_lines.append((line_num, obj))
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON parse error - {e}")
    
    print(f"\nResults:")
    print(f"Total lines: {total_lines}")
    print(f"Valid lines: {valid_lines}")
    print(f"Success rate: {valid_lines/total_lines*100:.1f}%" if total_lines > 0 else "  No lines")
    
    # Display samples
    if sample_lines:
        print(f"\nFirst few samples:")
        for line_num, obj in sample_lines:
            print(f"\nLine {line_num}:")
            print(f"conversation_id: {obj.get('conversation_id')}")
            print(f"task_id: {obj.get('task_id')}")
            print(f"Collection: {obj.get('Collection')}")
            print(f"input speakers: {[msg.get('speaker') for msg in obj.get('input', [])]}")
            print(f"contexts count: {len(obj.get('contexts', []))}")
            if obj.get('contexts'):
                first_ctx = obj['contexts'][0]
                print(f"first context: doc_id={first_ctx.get('document_id')}, score={first_ctx.get('score'):.2f}")
    
    print(f"\n{'='*70}\n")
    return valid_lines == total_lines


if __name__ == '__main__':
    test_files = [
        'results/clapnq_questions_submission.jsonl',
        'results/cloud_questions_submission.jsonl',
        'results/fiqa_questions_submission.jsonl',
        'results/govt_questions_submission.jsonl',
    ]
    
    results_dir = Path(__file__).parent / 'results'
    
    if results_dir.exists():
        test_files = list(results_dir.glob('*_submission.jsonl'))
    
    if not test_files:
        print("No output files found. Run the main pipeline first to generate results.")

    else:
        all_valid = True
        for output_file in sorted(test_files):
            if not test_output_format(output_file):
                all_valid = False
        
        if all_valid:
            print("All output files have valid format!")
        else:
            print("Some output files have format issues")
