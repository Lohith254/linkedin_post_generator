from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Settings
model_id = "microsoft/phi-2"
data_path = "./data/formatted_finetune_data.jsonl"
output_dir = "./model/lora-phi2-linkedin"

# Load dataset
dataset = load_dataset("json", data_files=data_path, split="train")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}  # Force CPU
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Tokenize dataset
def tokenize(entry):
    return tokenizer(entry["text"], truncation=True, padding="max_length", max_length=512)

tokenized_data = dataset.map(tokenize)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    logging_steps=10,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    save_total_limit=1,
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
if __name__ == "__main__":
    model.config.use_cache = False  # Required for LoRA training
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)