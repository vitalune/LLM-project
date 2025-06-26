# LLM-Project

This repository documents my step-by-step journey toward building and fine-tuning a large language model (LLM) from scratch, inspired by the book:

> **Raschka, Sebastian. _Build A Large Language Model (From Scratch)_. Manning, 2024. ISBN: 978-1633437166**

The project is organized into chapters, each corresponding to a Jupyter Notebook or script that focuses on a key stage in the LLM development pipeline- from tokenization and embeddings to transformer architecture, training, evaluation, and user interaction.

---

## 🔧 Project Overview

This LLM implementation is based on a custom-built GPT-style transformer, trained and fine-tuned using PyTorch. My version includes:

- 🧱 **Custom model architecture** (`GPTModel`) with positional and token embeddings, transformer decoder blocks, and a causal attention mechanism.
- 📜 **Tokenizer integration** using `tiktoken`, with support for converting between text and token IDs.
- 🧠 **Finetuning on instruction-following data**, resulting in the checkpoint `gpt2-medium355M-sft.pth` (not pushed due to GitHub size limits).
- 🧪 **Training and validation loss tracking** to monitor performance and generalization.
- 💬 **Interactive Chainlit UI** for running the model in a ChatGPT-style interface.
- ⚙️ **Model loading utilities** for inference using your own `.pth` weights.

---

## 📁 Repository Structure
LLM-project/
├── chapter02/         # Text preprocessing, tokenization, and embeddings
├── chapter03/         # Attention mechanisms and decoder-only transformer blocks
├── chapter04/         # Full GPT-style model implementation (pre-training setup)
├── chapter05/         # Pretraining with cross-entropy loss, loss tracking
├── finetuning/        # Supervised finetuning for text classification
├── instructions/
│   ├── instructions.ipynb  # Instruction-tuned finetuning (e.g., Alpaca-style)
│   └── evaluation.ipynb    # LLaMA-3.2B-based evaluation of outputs
├── ui/                # Chainlit-based local chat interface
│   ├── app.py
│   └── model/         # Contains GPTModel, generate(), tokenizer utils, config
│       ├── gpt_model.py
│       ├── generate.py
│       ├── tokenizer_utils.py
│       └── model_config.py
├── gpt2-medium355M-sft.pth  # Finetuned model weights (not pushed to GitHub)
└── README.md
---

## 💡 Project Summary

The goal of this project is to understand and implement all major components of a transformer-based language model. From raw text preprocessing to training on unlabeled data, and ultimately instruction-tuning, this repo captures the full LLM development pipeline.

### Key Accomplishments

- 🧱 Implemented a full decoder-only GPT architecture
- 📊 Trained and validated the model using cross-entropy loss
- 📎 Fine-tuned on supervised and instruction-following datasets
- 💬 Deployed a real-time chat interface using Chainlit
- 🔍 Evaluated outputs using LLaMA 3.2B and discussed limitations

---

## 📈 Training and Evaluation
	•	Training loss and validation loss are computed via cross-entropy, comparing predicted token distributions to actual next tokens.
	•	The project tracks loss over time to monitor generalization and overfitting.
	•	Later-stage evaluation is done using instruction-based prompts and judged with LLaMA 3.2B, though stronger models (e.g. OpenChat or Mistral) are recommended for more reliable scoring.

---

## Future Goals
* Increase weight parameters and size.
* Use newer and more efficient algorithms to build model architecture.
* Connect with interested companies and individuals to enhance outdated systems.

## Contact

Feel free to open an issue or discussion if you want to contribute, ask questions, or build on this work! Also feel free to contact me via email at amirvalizadeh161@gmail.com.

## Citation

Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024.
ISBN: 978-1633437166
