from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def load_finetuned_model():
    base_model_id = "microsoft/phi-2"
    lora_model_path = "./model/lora-phi2-linkedin"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    return model, tokenizer

def clean_output(text):
    # Remove everything before "ANSWER:" or "OUTPUT:" or similar markers
    for marker in ["ANSWER:", "##OUTPUT", "###OUTPUT", "## OUTPUT", "### OUTPUT"]:
        if marker in text:
            text = text.split(marker, 1)[-1]

    # Remove any leftover prompt/markdown like "###", "##", or "INPUT"
    lines = text.strip().splitlines()
    cleaned_lines = [
        line.strip() for line in lines
        if not line.strip().lower().startswith(("input", "##", "###", "output"))
    ]

    #print("\n".join(cleaned_lines).strip())
    return "\n".join(cleaned_lines).strip()

def generate_post(prompt, max_tokens=150):
    model, tokenizer = load_finetuned_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text = clean_output(generated_text)
    return cleaned_text 

# Test LLM
if __name__ == "__main__":
    prompt = "Write a LinkedIn post announcing a new job at Microsoft as a Data Analyst."
    post = generate_post(prompt)
    print("\n📢 Generated LinkedIn Post:\n")
    print(post)