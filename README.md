# LLM Fine-Tuning, Alignment, and Deployment with Unsloth

## Project Overview
This repository contains code, Colab notebooks, and video tutorials (placeholder) demonstrating various techniques for fine-tuning, aligning, and deploying large language models (LLMs) using the **Unsloth** library for high-performance training.

The project covers the following key workflows (corresponding to assignment parts A-G):
- **A: Fine-tuning** open-weight LLMs for specific tasks (e.g., coding instruction following with Llama 3.1).
- **B: Continued pretraining** to adapt models to new languages (e.g., teaching TinyLlama Hindi).
- **C: Advanced Chat Template Applications** including classification, conversational chat, extending TinyLlama's context window, and fine-tuning on multiple datasets simultaneously.
- **D: Reward modeling** using **ORPO** and **DPO** for preference alignment (using Phi-3 Mini).
- **E: Continued fine-tuning** starting from previously saved custom LoRA checkpoints.
- **F: Mental health chatbot development** via fine-tuning Phi-3 Mini, with important ethical considerations.
- **G: Model export to GGUF** format for local inference with **Ollama**.

---

## Models Demonstrated
This project primarily demonstrates techniques using the following models fine-tuned with **Unsloth**:

-   **Llama 3 / Llama 3.1 (8B)**: Used for instruction fine-tuning (Part A) and export to Ollama (Part G). Known for strong general language capabilities.
-   **TinyLlama (1.1B)**: Used for continued pretraining (Part B), advanced chat template examples (Part C), and continued fine-tuning from checkpoints (Part E) due to its small size and speed.
-   **Phi-3 Mini (3.8B)**: Used for reward modeling (DPO/ORPO, Part D) and the mental health chatbot (Part F) because of its balance between capability and efficiency.

*Note: The techniques shown (fine-tuning, LoRA, DPO, ORPO, export) can often be applied to other compatible models listed in the Unsloth documentation with appropriate adjustments.*

---

## Colab Notebooks & Demonstrations
Click the links below to open the Google Colab notebooks for each part of the assignment:

*   **A: Fine-Tuning (Llama 3.1 - Coding Task)**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lSrjHD4ETKGEdDFM3Kx55hph1oUq-mTf?usp=sharing)
*   **B: Continued Pretraining (TinyLlama - Hindi)**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dwbOaaTijWD4dFQQWTNpfcoHTAZRqAtg?usp=sharing)
*   **C: Chat Template Applications (TinyLlama)**
    *   (Includes Classification, Conversational Chat, Extended Context, Multi-Dataset Finetune)
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GCmozNN8ar6ipVFY8QT-QiFIyMTz_QKZ?usp=sharing)
*   **D: Reward Modeling (ORPO & DPO with Phi-3 Mini)**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13l6804kN7Gti8oOUD1lL_IkYiAJW8w0E?usp=sharing)
*   **E: Continued Fine-Tuning from Checkpoint (TinyLlama)**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/131U-JK0XNNCh1l6wtIJj6sTJM-kM72Um?usp=sharing)
*   **F: Mental Health Chatbot (Phi-3 Mini)**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mf3NYuTv92dr9V0LY3Y8P5PjyoIJk5sn?usp=sharing)
*   **G: Exporting to Ollama (Llama 3)**
    *   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CZjY2Zm7mYc8huqjWAP-LZBlMf6NoAmx?usp=sharing)

---

## Key Features Demonstrated

### Fine-Tuning with Unsloth (Part A)
-   **Efficient Fine-tuning** using **QLoRA** adapters with Unsloth optimizations.
-   Uses **4-bit quantization** for significant memory reduction.
-   Demonstrated instruction following for coding tasks with Llama 3.1.

### Continued Pretraining (Part B)
-   **Adapting LLMs to New Languages** by continued pretraining on raw text corpora (e.g., Hindi).
-   Improves vocabulary and pattern recognition for the target language (demonstrated with TinyLlama).

### Advanced Chat Templates (Part C)
-   Utilizing chat templates for various tasks beyond simple conversation:
    -   Framing **Classification** as an instruction task.
    -   Handling multi-turn **Conversational Chat**.
    -   **Extending Max Context Size** for models like TinyLlama using Unsloth's features.
    -   Combining **Multiple Datasets** (e.g., chat and code) into a single fine-tuning run.

### Reward Modeling (Part D)
-   Aligning model preferences using state-of-the-art techniques:
    -   **DPO (Direct Preference Optimization)** using preference pairs.
    -   **ORPO (Odds Ratio Preference Optimization)** combining LM loss with preference optimization.
    -   Demonstrated with Phi-3 Mini on the Ultrafeedback dataset.

### Continued Fine-Tuning (Part E)
-   Loading previously saved **LoRA adapter checkpoints**.
-   Continuing the fine-tuning process, allowing for iterative improvement or adaptation.

### Mental Health Chatbot (Part F)
-   Fine-tuning **Phi-3 Mini** on the `Amod/mental_health_counseling_conversations` dataset.
-   Focuses on generating empathetic and supportive responses.
-   **Includes critical ethical considerations and disclaimers** in the system prompt.

### Ollama Export (Part G)
-   Merging fine-tuned LoRA adapters into the base model.
-   **Exporting the merged model** to **GGUF format** using Unsloth's `save_pretrained_gguf`.
-   Provides instructions for creating an Ollama `Modelfile` and running the model locally.

---

## Video Tutorials
Watch the video walkthroughs for each part:
[![YouTube Playlist](https://img.shields.io/badge/YouTube-Watch_Tutorials-red?logo=youtube)](https://youtube.com/playlist?list=YOUR_PLAYLIST_ID_HERE)

---

## Usage Workflow Overview

### 1️⃣ Fine-Tuning LLMs (Part A)
-   Select a base model (e.g., Llama 3.1 Instruct).
-   Prepare instruction-following data (e.g., coding).
-   Configure and run SFTTrainer with Unsloth + QLoRA.

### 2️⃣ Continued Pretraining (Part B)
-   Select a base model (e.g., TinyLlama Base).
-   Load a raw text dataset in the target language.
-   Run SFTTrainer with `packing=True` for pretraining objective.

### 3️⃣ Chat Templates (Part C)
-   Load base model (e.g., TinyLlama).
-   Define formatting functions using `tokenizer.apply_chat_template`.
-   Prepare datasets for classification, chat, long context, or multi-task scenarios.
-   Fine-tune using SFTTrainer.

### 4️⃣ Reward Modeling (Part D)
-   Load a base instruction-tuned model (e.g., Phi-3 Mini Instruct).
-   Load a preference dataset (e.g., Ultrafeedback binarized).
-   Prepare data (often requires formatting chosen/rejected responses).
-   Configure and run `DPOTrainer` or `ORPOTrainer`.

### 5️⃣ Continued Fine-Tuning (Part E)
-   Run an initial fine-tuning phase (like Part A or C) and save adapters.
-   Reload the original base model.
-   Re-apply the *exact same* LoRA config structure.
-   Load the saved adapter weights using `PeftModel.from_pretrained`.
-   Configure SFTTrainer and continue training.

### 6️⃣ Mental Health Chatbot (Part F)
-   Select a suitable base model (e.g., Phi-3 Mini Instruct).
-   Load relevant counseling/support dataset.
-   Define a formatting function with a strong ethical system prompt and disclaimers.
-   Fine-tune using SFTTrainer. Test responses carefully.

### 7️⃣ Export to Ollama (Part G)
-   Perform fine-tuning (like Part A).
-   Merge LoRA adapters using `model.merge_and_unload()`.
-   Export to GGUF using `model.save_pretrained_gguf()`.
-   Download GGUF, create `Modelfile`, and use Ollama commands locally.

---

## Tips & Best Practices
-   **Optimize Memory:** Use **QLoRA** and **4-bit quantization** via Unsloth. Use gradient checkpointing (`use_gradient_checkpointing="unsloth"` or `True`). Manage batch size and gradient accumulation.
-   **Dataset Quality:** Clean, diverse, and high-quality datasets are crucial for good fine-tuning results.
-   **Chat Templates:** Ensure your data formatting strictly adheres to the target model's expected chat template (e.g., ChatML for Phi-3, Llama 3 format for Llama 3).
-   **Reward Modeling LRs:** Note that ORPO often requires significantly lower learning rates than DPO or SFT.
-   **Merging & Export:** Merging adapters requires substantial RAM. Exporting might create a directory; ensure you grab the correct `.gguf` file from within it.
-   **Ethical AI:** Be especially mindful when fine-tuning for sensitive applications like mental health. Prioritize safety, disclaimers, and user well-being. Never deploy without rigorous testing and safety measures.

## References
-   [Unsloth Documentation](https://docs.unsloth.ai)
-   [Unsloth DPO/ORPO Docs](https://docs.unsloth.ai/basics/reward-modelling-dpo-and-orpo)
-   [Unsloth Ollama Export Tutorial](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)
-   [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook)
-   [Mental Health Chatbot Article (Hint)](https://medium.com/@mauryaanoop3/fine-tuning-microsoft-phi3-with-unsloth-for-mental-health-chatbot-development-ddea4e0c46e7)
-   [Ollama](https://ollama.com/)
