---
title: "Introduction to LLM Fine-Tuning"
difficulty: intermediate
topic: llm
order: 8
estimatedTime: "15 minutes"
summary: "Introduces LLM fine-tuning concepts including full fine-tuning vs. parameter-efficient methods like LoRA and QLoRA, with practical guidance on dataset preparation and training setup."
---
# Introduction to LLM Fine-Tuning

## Overview

Pre-trained LLMs are general-purpose text generators. Fine-tuning adapts them to specific tasks, domains, or styles by continuing training on a curated dataset. The central question practitioners face is: **when should you fine-tune, and when is prompt engineering enough?**

Prompt engineering is the right choice when you need quick iteration, have no labeled data, or your task is well-served by in-context examples. Fine-tuning becomes necessary when you need consistent style or formatting that prompts cannot reliably enforce, when you want to internalize domain knowledge (medical, legal, financial), when latency matters and you want to avoid long system prompts, or when you need the model to learn behaviors that are difficult to describe in natural language.

## Full Fine-Tuning vs. Parameter-Efficient Methods

**Full fine-tuning** updates every parameter in the model. For a 7B-parameter model stored in fp16, this requires roughly 14 GB just for weights, plus optimizer states (Adam stores two additional copies), gradient tensors, and activations -- easily exceeding 100 GB of GPU memory. Full fine-tuning delivers maximum expressiveness but is expensive and risks catastrophic forgetting of the base model's general capabilities.

**Parameter-efficient fine-tuning (PEFT)** methods freeze most of the base model and train only a small number of additional or modified parameters. The most popular approaches are:

- **LoRA (Low-Rank Adaptation):** Injects trainable low-rank matrices into attention layers.
- **QLoRA:** Combines LoRA with 4-bit quantization of the base model, drastically reducing memory.
- **Adapters:** Small bottleneck layers inserted between existing transformer blocks.
- **Prefix Tuning:** Prepends trainable "virtual tokens" to the input at each layer.

## The Math Behind LoRA

LoRA's key insight is that weight updates during fine-tuning have low intrinsic rank. Instead of updating a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ directly, LoRA decomposes the update into two low-rank matrices:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.

The number of trainable parameters drops from $d \times k$ to $r \times (d + k)$. For a typical attention projection with $d = k = 4096$ and $r = 16$, this is a reduction from 16.8M parameters to 131K -- a **128x reduction**.

During inference, the product $BA$ can be merged back into $W_0$, so there is **zero additional latency** compared to the original model. The forward pass for an input $x$ becomes:

$$h = W_0 x + BAx = W_0 x + B(Ax)$$

The scaling factor $\alpha / r$ is applied to the LoRA output to control the magnitude of the adaptation relative to the pre-trained weights.

## Dataset Preparation

Fine-tuning datasets typically follow an instruction format:

1. **Instruction-Response pairs:** Each example has an instruction (or prompt) and a desired completion.
2. **Quality over quantity:** A few thousand high-quality examples often outperform millions of noisy ones.
3. **Formatting consistency:** Use a consistent chat template (e.g., ChatML, Alpaca format).
4. **Decontamination:** Remove examples that overlap with your evaluation set.

Common dataset sizes: 1K-10K examples for style/format adaptation, 10K-100K for domain specialization, 100K+ for teaching entirely new capabilities.

## Training Setup with PEFT and LoRA

Below is a practical example using HuggingFace's PEFT library to set up LoRA fine-tuning:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Load base model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # QLoRA: 4-bit quantization
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # Rank of the low-rank matrices
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.05,      # Dropout for regularization
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

# Initialize trainer (assumes `dataset` is a HuggingFace Dataset object)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)

trainer.train()
```

## Evaluation

After fine-tuning, evaluate on held-out data using:

- **Perplexity:** Measures how well the model predicts the next token on test data. Lower is better.
- **Task-specific metrics:** BLEU, ROUGE, F1, or accuracy depending on your use case.
- **Human evaluation:** For subjective quality (style, helpfulness, safety).
- **Overfitting checks:** Compare train loss vs. validation loss. If validation loss increases while train loss decreases, reduce epochs or increase regularization.

## Key Takeaways

1. Start with prompt engineering; fine-tune only when prompting hits a ceiling.
2. LoRA and QLoRA make fine-tuning accessible on consumer GPUs (a 7B model can be fine-tuned on a single 24 GB card with QLoRA).
3. Data quality dominates data quantity -- invest in curation.
4. Always merge LoRA weights for production deployment to avoid inference overhead.
5. Monitor for catastrophic forgetting by evaluating on general benchmarks alongside your target task.
