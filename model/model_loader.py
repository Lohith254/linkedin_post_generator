from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_phi2_model():
    """
    Loads the Phi-2 model and tokenizer from Hugging Face.
    Uses automatic device allocation (CPU or GPU).
    """
    model_id = "microsoft/phi-2"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with automatic device map and half precision if CUDA available
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16  # use half precision on GPU
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    return model, tokenizer

# Quick test
if __name__ == "__main__":
    model, tokenizer = load_phi2_model()
    print("✅ Phi-2 model and tokenizer loaded successfully.")