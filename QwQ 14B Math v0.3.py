#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
#   <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/u54VK8m8tk"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
#   <a href="https://ko-fi.com/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
#
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions).
#
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save) (eg for Llama.cpp).
#
# Features in the notebook:
# 1. Uses Maxime Labonne's [FineTome 100K](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset.
# 1. Convert ShareGPT to HuggingFace format via `standardize_sharegpt`
# 2. Train on Completions / Assistant only via `train_on_responses_only`
# 3. Unsloth now supports all Torch, all TRL & Xformers versions & Python 3.12!

# ## Kaggle is slow - you'll have to wait **5 minutes** for it to install.
#
# I suggest you to use our free Colab notebooks instead. I linked our Llama 3.1 8b Colab notebook here: [notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)

# In[1]:

import os


os.system('pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"')
print("Unsloth installed.")
os.system('pip install wandb')
print("WandB installed.")


# * We support Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen, TinyLlama, Vicuna, Open Hermes etc
# * We support 16bit LoRA or 4bit QLoRA. Both 2x faster.
# * `max_seq_length` can be set to anything, since we do automatic RoPE Scaling via [kaiokendev's](https://kaiokendev.github.io/til) method.
# * [**NEW**] We make Llama-3.2 1/3B **2x faster**! See our [Llama-3.2 notebook](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)
# * [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)

# ### Config
#
#

# In[ ]:


model_name = "unsloth/Qwen2.5-14B"
max_seq_length = 8192
load_in_4bit = False
dtype = None
lora_rank = 256
rank_stabilized = True
hf_token = "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"
lora_repo = None #"qingy2024/OwO-14B-LoRA"
merge_repo = "qingy2024/QwQ-Math-14B-v0.3"

# ---< Training >----------------------------------------

batch_size = 4
gradient_acc = 2
max_steps = -1 # or -x where x is the epochs, so -5 is 5 epochs.
seed = 3407
lr_scheduler = "cosine"
learning_rate = 2e-4


# ---

# In[2]:


from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[3]:


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = seed,
    use_rslora = rank_stabilized,
    loftq_config = None,
)


# <a name="Data"></a>
# ### Data Prep
# We now use the `Qwen-2.5` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. But we convert it to HuggingFace's normal multiturn format `("role", "content")` instead of `("from", "value")`/ Qwen renders multi turn conversations like below:
#
# ```
# <|im_start|>system
# You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
# <|im_start|>user
# What is 2+2?<|im_end|>
# <|im_start|>assistant
# It's 4.<|im_end|>
#
# ```
#
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3` and more.

# In[4]:


from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("qingy2024/QwQ-LongCoT-Verified-130K", "sharegpt-verified-cleaned", split="train")
dataset = dataset.map(apply_template, batched=True)


# In[ ]:


import os
os.environ['WANDB_API_KEY'] = '0aee5395a94fbd8e33ada07b71309b1f30561cac'


# In[5]:


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_acc,

        num_train_epochs = 1, # Full epoch

        # max_steps=max_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed = seed,
        warmup_steps=10,
        output_dir="output",
        report_to = "wandb",
    ),
)


# In[16]:


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[17]:


trainer_stats = trainer.train()


# In[ ]:


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can change the instruction and input - leave the output blank!
#
# **[NEW] Try 2x faster inference in a free Colab for Llama-3.1 8b Instruct [here](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)**
#
# We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
#
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[19]:


if lora_repo is not None:
    model.push_to_hub(lora_repo, token = hf_token)
    tokenizer.push_to_hub(lora_repo, token = hf_token)

if merge_repo is not None:
    model.push_to_hub_merged(merge_repo, tokenizer, save_method = "merged_16bit", token = hf_token)
