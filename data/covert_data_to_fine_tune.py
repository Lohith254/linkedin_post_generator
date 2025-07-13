import json

input_file = "data/linkedin_posts.jsonl"  # original file
output_file = "data/formatted_finetune_data.jsonl"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        example = json.loads(line)
        prompt = example["prompt"]
        completion = example["completion"]
        combined = f"### Prompt: {prompt}\n### Completion: {completion}"
        fout.write(json.dumps({"text": combined}) + "\n")

print(f"✅ Saved fine-tuning-ready file to {output_file}")