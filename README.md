
# 📝 LinkedIn Post Generator (Fine-Tuned LLM with Streamlit UI)

Generate professional and engaging LinkedIn posts using a fine-tuned language model (LoRA-based). Whether it's a new job announcement, promotion, or appreciation post, this tool turns prompts into polished LinkedIn-ready content.

---

## 🚀 Features

- 🔧 Fine-tuned LLM using **LoRA** for LinkedIn-style generation.
- 🖥️ Simple **Streamlit UI** for interactive post creation.
- 📥 Input flexible prompts like: _"Write a post about my promotion to Senior Data Scientist"_
- ✨ Outputs only the final LinkedIn post (clean and professional).
- 📁 Easily extensible for new templates and fine-tuning.

---

## 🛠️ How It Works

1. **Training**:  
   A pre-trained LLM (e.g., LLaMA or GPT) is fine-tuned using LoRA adapters on a dataset of LinkedIn-style prompts and outputs.

2. **UI**:  
   Users enter their prompt via a Streamlit app, and the model returns a clean, ready-to-use LinkedIn post.

---

## 📦 Folder Structure

```
linkedIn_llm/
├── app.py                  # Streamlit UI
├── generate_post.py        # Code to generate post from prompt
├── train_lora.py           # LoRA fine-tuning script
├── model/                  # Fine-tuned model weights
├── data/                   # Training data (optional)
└── README.md               # This file
```

---

## 🔧 Setup

```bash
# Clone the repo
git clone https://github.com/Lohith254/linkedin_post_generator.git
cd linkedin_post_generator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## ✍️ Example Usage

**Prompt:**
```
Write a professional LinkedIn post about getting a new job at Microsoft as Marketing Manager.
```

**Output:**
```
Excited to announce my new role as Marketing Manager at Microsoft! Grateful for this opportunity and looking forward to the journey ahead. #LinkedIn #Marketing #NewJob
```

---

## 🧠 Model Details

- Base Model: `Meta/LLaMA` or other transformer-based causal LLM.
- Fine-Tuning: LoRA adapters via `peft` and `transformers`.
- Frameworks: `PyTorch`, `HuggingFace`, `Streamlit`

---

## 🪪 License

This project is licensed under the MIT License.  
Feel free to use and modify it for your own applications!

---

## 💡 Future Improvements

- Add multiple LinkedIn post categories.
- Improve prompt understanding with classification.
- Deploy to HuggingFace or Streamlit Cloud.

---

## 🙌 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co)
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
- [Streamlit](https://streamlit.io)

---

📬 Have ideas to collaborate or improve this? Open a pull request or issue!
