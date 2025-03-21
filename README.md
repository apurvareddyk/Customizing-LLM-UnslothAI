# LLM Fine-Tuning with Unsloth

## Project Overview
This repository contains code, Colab notebooks, and video tutorials for fine-tuning and deploying large language models (LLMs) using **Unsloth**. The project covers:
- **Fine-tuning** open-weight LLMs for various use cases (coding, chat, etc.)
- **Continued pretraining** for new language acquisition
- **Chat templates** for different tasks (classification, conversational chat, max context extension)
- **Reward modeling** with ORPO and DPO
- **Continued fine-tuning from custom checkpoints**
- **Model export to Ollama** for inference and deployment
- **Mental health chatbot development** using Phi-3 fine-tuning

---

## Models Used
This project explores multiple LLMs with **Unsloth** fine-tuning, covering different sizes and architectures:

- **Llama 3.1 (8B)** – General-purpose language model optimized for chat and text generation.  
- **Mistral NeMo (12B)** – Suitable for large-scale language generation and multi-turn conversation.  
- **Gemma 2 (9B)** – Optimized for reasoning and complex NLP tasks.  
- **Inference Chat UI** – For interactive, real-time chat applications.  
- **Phi-3.5 (mini)** – Small but powerful, designed for efficient inference.  
- **Llama 3 (8B)** – Compact model with strong language capabilities.  
- **Mistral v0.3 (7B)** – Balanced model with moderate size and fast inference.  
- **Phi-3 (medium)** – Mid-range model offering a balance between size and performance.  
- **Qwen2 (7B)** – Capable of handling multilingual text generation and NLP tasks.  
- **Gemma 2 (2B)** – Lightweight model for rapid inference.  
- **TinyLlama** – Extremely small model designed for low-latency applications.  

---

## Colab Notebooks
Click below to open the Colab notebooks:

- [Fine-Tuning LLMs with Unsloth](https://colab.research.google.com/)  
- [Continued Pretraining](https://colab.research.google.com/)  
- [Chat Templates](https://colab.research.google.com/)  
- [Reward Modeling with ORPO & DPO](https://colab.research.google.com/)  
- [Continued Fine-Tuning from Checkpoint](https://colab.research.google.com/)  
- [Exporting to Ollama](https://colab.research.google.com/)  
- [Mental Health Chatbot](https://colab.research.google.com/)  

---

## Key Features

### Fine-Tuning with Unsloth
- **Efficient LLaMA fine-tuning** with **QLoRA** adapters.
- Uses **4-bit quantization** for memory efficiency.
- Supports multiple chat templates and inference formats.

### Continued Pretraining
- **Expand language capabilities** by continued pretraining on new language datasets.
- Improves vocabulary adaptation and contextual understanding.

### Chat Templates
- **Custom chat templates** for different use cases:
    - **Classification** tasks  
    - **Conversational chat** scenarios  
    - **Max context size extension** for TinyLlama  

### Reward Modeling
- Fine-tune reward models with **ORPO (Optimal Response Policy Optimization)** and **DPO (Direct Preference Optimization)**.
- Improve task-specific generation quality.

### Ollama Export
- **Export fine-tuned models** to Ollama for efficient, local inference.
- Demonstrates practical deployment workflows.

### Mental Health Chatbot
- Fine-tuning **Phi-3** on mental health-related datasets.
- Generates contextually sensitive and supportive responses.

---

## Video Tutorials
Watch the entire series on YouTube:  
[![YouTube Playlist](https://img.shields.io/badge/YouTube-Watch-red?logo=youtube)](https://youtube.com)

---

## Usage Workflow

### 1️⃣ Fine-Tuning LLMs
- Use **QLoRA** with **Unsloth** for fine-tuning various LLMs.
- Specify the model size, chat template, and dataset.
- Export the fine-tuned models for inference.

### 2️⃣ Continued Pretraining
- Load new language datasets.
- Use Unsloth's **continued pretraining** feature.
- Monitor the model’s improved performance on the new language.

### 3️⃣ Chat Templates
- Define multiple templates for classification, conversation, and max context extension.
- Experiment with different formats and parameters.

### 4️⃣ Reward Modeling
- Use **ORPO** and **DPO** methods to fine-tune reward models.
- Improve response quality for specific tasks.

### 5️⃣ Export to Ollama
- Convert the fine-tuned models into **Ollama-compatible** format.
- Deploy locally for fast inference.

---

## Tips & Best Practices
- **Optimize Memory Usage:** Use **QLoRA** and **4-bit quantization** to reduce VRAM requirements.  
- **Dataset Curation:** Use clean, high-quality datasets for fine-tuning to ensure better generalization.  
- **Template Variation:** Experiment with multiple **chat templates** to see how different prompts affect outputs.  
- **Inference Speed:** Use **Ollama export** for efficient local inference.  

## References
- [Unsloth Documentation](https://docs.unsloth.ai)  
- [ORPO & DPO Reward Modeling](https://docs.unsloth.ai/basics/reward-modelling-dpo-and-orpo)  
- [Exporting to Ollama](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)  
- [Mental Health Chatbot Development](https://medium.com/@mauryaanoop3/fine-tuning-microsoft-phi3-with-unsloth-for-mental-health-chatbot-development-ddea4e0c46e7)  

