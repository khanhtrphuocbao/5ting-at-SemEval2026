import json

with open('human/evaluations/RAG.json') as f:
    data = json.load(f)

# Check tasks structure
if 'tasks' in data:
    print(f"Total tasks: {len(data['tasks'])}")
    if data['tasks']:
        first_task = data['tasks'][0]
        print(f"\nFirst task keys: {list(first_task.keys())}")
        print(f"First task: {json.dumps(first_task, indent=2)[:500]}")

# Check documents structure  
if 'documents' in data:
    print(f"\n\nTotal documents: {len(data['documents'])}")
    if data['documents']:
        first_doc = data['documents'][0]
        print(f"First document keys: {list(first_doc.keys())}")
        print(f"First document: {json.dumps(first_doc, indent=2)[:500]}")
