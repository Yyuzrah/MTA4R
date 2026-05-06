# src/train_flexible.py
import argparse
import json
import os
import random
import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers import Trainer, TrainingArguments
import wandb

def main(args):
    # 1. Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Strictly set random seeds for reproducibility
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    accelerator = Accelerator()
    
    # Weights & Biases (W&B) setup
    if args.wandb and accelerator.is_main_process:
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"], # WARNING: Ensure this is anonymized in the config file!
            name=config["name"],
            config=config,
            dir=config["wandb"]["dir"],
        )

    print(f"Constructing Model for RIGOROUS comparison: Architecture={config['arch_type']}, PE={config['pe_type']}")
    
    VOCAB_SIZE = 50257 
    
    if config["pe_type"] == "rope":
        # RoPE -> GPT-NeoX
        model_config = GPTNeoXConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=config["n_embd"],
            num_hidden_layers=config["n_layer"],
            num_attention_heads=config["n_head"],
            intermediate_size=config["n_embd"] * 4,
            hidden_act="gelu",
            max_position_embeddings=config["context_length"],
            rotary_pct=1.0,
            use_parallel_residual=False,
            attention_bias=True,
            use_cache=False if config.get("gradient_checkpointing") else True
        )
        if torch.cuda.get_device_capability()[0] >= 8:
            model_config.attn_implementation = "flash_attention_2"
            
        model = GPTNeoXForCausalLM(model_config)
    
    else: # ape
        # APE -> GPT-Neo
        # [FINAL FIX]: attention_types must be in the format [[["type"], num]]
        # Note the brackets around ["global"]! This ensures only one element is added 
        # during 'extend', preventing string unpacking.
        
        correct_attention_types = [[["global"], config["n_layer"]]]
        
        model_config = GPTNeoConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=config["n_embd"],
            num_layers=config["n_layer"],
            num_heads=config["n_head"],
            intermediate_size=config["n_embd"] * 4,
            hidden_act="gelu",
            max_position_embeddings=config["context_length"],
            
            # The format here must strictly match: [[[types], count]]
            attention_types=correct_attention_types,
            
            window_size=256, # Placeholder
            use_cache=False if config.get("gradient_checkpointing") else True
        )
        model = GPTNeoForCausalLM(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized. Total Parameters: {num_params/1e6:.2f}M")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config["context_length"]

    train_file = os.path.join(config["data_dir"], config["train_file"])
    val_file = os.path.join(config["data_dir"], config["val_file"])
    
    print(f"Loading FULL datasets from: {train_file}")
    hf_datasets = load_dataset("json", data_files={"train": train_file, "val": val_file})
    
    print(f"Training Samples: {len(hf_datasets['train'])}")
    print(f"Validation Samples: {len(hf_datasets['val'])}")

    def tokenize(element):
        text = [
            tokenizer.bos_token + element["search_path"][e].strip() + tokenizer.eos_token
            for e in range(len(element["search_path"]))
        ]
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=config["context_length"],
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
    )
    
    def collate_fn(examples):
        batch = tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    # Compatibility for eval_strategy (Transformers version handling)
    eval_strategy_kw = "eval_strategy"
    import transformers
    if transformers.__version__ < "4.41.0":
        eval_strategy_kw = "evaluation_strategy"

    training_args_dict = {
        "output_dir": config["output_dir"],
        "per_device_train_batch_size": config["batch_size"],
        eval_strategy_kw: "steps",
        "eval_steps": config["eval_steps"],
        "logging_steps": config["log_steps"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "gradient_checkpointing": config.get("gradient_checkpointing", False),
        "num_train_epochs": config["num_train_epochs"],
        "weight_decay": config["weight_decay"],
        "warmup_steps": config["warmup_steps"],
        "learning_rate": config["lr"],
        "save_strategy": "steps",
        "save_total_limit": config["save_total_limit"],
        "save_steps": config["save_steps"],
        "seed": config["seed"],
        "bf16": True,
        "report_to": "wandb" if args.wandb else "none",
        "run_name": config["name"],
        "ddp_find_unused_parameters": False,
    }

    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(os.path.join(config["output_dir"], "final_model"))
    
    with open(os.path.join(config["output_dir"], "final_model", "exp_config.json"), "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)