#!/usr/bin/env python3
"""
GPT-OSS-20B Fine-tuning Script for Customer Support
Optimized for high-memory GPUs (80GB+ recommended)
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_ID = "openai/gpt-oss-20b"
OUTPUT_DIR = "./gpt-oss-20b-customer-support"
DATASET_NAME = "Kaludi/Customer-Support-Responses"

def main():
    print("=== GPT-OSS-20B Customer Support Fine-tuning ===")
    
    # Check GPU availability and memory
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 60:
            print("⚠️  Warning: GPU memory < 60GB. This model requires high memory GPU (80GB+ recommended)")
    
    # 1. Load customer support dataset
    print("\n1. Loading dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"Sample: {dataset[0]}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # 2. Convert to instruction format
    print("\n2. Converting dataset format...")
    def format_example(example):
        return {
            "text": f"### Human: {example['query']}\n\n### Assistant: {example['response']}"
        }
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print("✅ Dataset formatted successfully")
    
    # 3. Create train/validation split
    splits = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = splits["train"], splits["test"]
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # 4. Load tokenizer
    print("\n3. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return
    
    # 5. Load model with memory optimizations
    print("\n4. Loading GPT-OSS-20B model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            max_memory={0: "75GB"},  # Reserve memory for training
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("💡 This usually means insufficient GPU memory. Try:")
        print("   - Use GPU with 80GB+ memory (A100, H100)")
        print("   - Reduce max_memory setting")
        print("   - Use multi-GPU setup")
        return
    
    # 6. Setup LoRA for efficient fine-tuning
    print("\n5. Setting up LoRA configuration...")
    try:
        lora_config = LoraConfig(
            r=16,                    # Rank for LoRA
            lora_alpha=32,           # Scaling parameter
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
            lora_dropout=0.1,        # Dropout for regularization
            bias="none",             # Don't train bias terms
            task_type="CAUSAL_LM",   # Task type
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✅ LoRA setup completed")
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        return
    
    # 7. Training configuration
    print("\n6. Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                    # Number of training epochs
        per_device_train_batch_size=2,         # Batch size per device
        per_device_eval_batch_size=1,          # Evaluation batch size
        gradient_accumulation_steps=8,         # Effective batch size = 2 * 8 = 16
        gradient_checkpointing=True,           # Memory optimization
        optim="paged_adamw_32bit",            # Memory-efficient optimizer
        logging_steps=10,                      # Log every 10 steps
        save_strategy="epoch",                 # Save after each epoch
        learning_rate=2e-4,                    # Learning rate
        bf16=True,                             # Use bfloat16 for efficiency
        max_grad_norm=0.3,                     # Gradient clipping
        warmup_ratio=0.03,                    # Warmup ratio
        lr_scheduler_type="cosine",            # Learning rate scheduler
        report_to="none",                      # Disable external logging
        save_total_limit=2,                    # Keep only last 2 checkpoints
        # Memory optimizations
        dataloader_pin_memory=False,           # Disable pin memory
        remove_unused_columns=False,           # Keep columns for LoRA
        max_seq_length=1024,                   # Maximum sequence length
        dataloader_num_workers=2,              # Number of data loader workers
    )
    
    # 8. Setup trainer
    print("\n7. Setting up SFT trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=1024,               # Maximum sequence length
            packing=False,                     # Disable packing for better control
            dataset_text_field="text",         # Text field in dataset
        )
        print("✅ Trainer setup completed")
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return
    
    # 9. Clear GPU cache before training
    print("\n8. Clearing GPU cache...")
    torch.cuda.empty_cache()
    
    # 10. Start training
    print("\n9. Starting fine-tuning...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save the fine-tuned model
        print("\n10. Saving fine-tuned model...")
        trainer.model.save_pretrained(f"{OUTPUT_DIR}-final")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
        print(f"✅ Model saved to {OUTPUT_DIR}-final")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 11. Test the fine-tuned model
    print("\n11. Testing fine-tuned model...")
    try:
        test_prompt = "### Human: My order arrived damaged. What should I do?\n\n### Assistant:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test prompt: My order arrived damaged. What should I do?")
        print(f"Model response: {response}")
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
    
    print("\n🎉 Fine-tuning process completed successfully!")
    print(f"📁 Model saved to: {OUTPUT_DIR}-final")
    print("💡 You can now use the fine-tuned model for customer support tasks!")

if __name__ == "__main__":
    main()
