import os
import json
import argparse


def trim_top_k_contexts(input_path: str, output_path: str, k: int = 5):
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	total_in = 0
	total_out = 0
	with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
		for lineno, line in enumerate(fin, start=1):
			line = line.strip()
			if not line:
				continue
			total_in += 1
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				# skip malformed lines but report
				print(f"Skipping malformed JSON on line {lineno}")
				continue

			if isinstance(obj, dict) and 'contexts' in obj and isinstance(obj['contexts'], list):
				obj['contexts'] = obj['contexts'][:k]

			fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
			total_out += 1

	print(f"Processed {total_in} input lines, wrote {total_out} output lines to {output_path}")


def main():
	parser = argparse.ArgumentParser(description='Create retrieval top-5 JSONL from A outputs')
	parser.add_argument('--input', '-i', default=os.path.join('subtask_B', 'data_from_A', '16_1_rewritre_lastturn_5_mini.jsonl'),
						help='Input JSONL file path')
	parser.add_argument('--output', '-o', default=os.path.join('subtask_B', 'A_top5', 'retrieval_top5.jsonl'),
						help='Output JSONL file path (will be created)')
	parser.add_argument('--k', '-k', type=int, default=5, help='Number of top contexts to keep')

	args = parser.parse_args()
	trim_top_k_contexts(args.input, args.output, args.k)


if __name__ == '__main__':
	main()

