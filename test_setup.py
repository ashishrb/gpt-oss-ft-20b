#!/usr/bin/env python3
"""
Test script to verify GPT-OSS-20B fine-tuning setup
Run this before attempting the full fine-tuning
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer

def test_setup():
    print("=== Testing GPT-OSS-20B Fine-tuning Setup ===\n")
    
    # Test 1: PyTorch and CUDA
    print("1. Testing PyTorch and CUDA...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check if memory is sufficient
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 60:
            print(f"   ⚠️  Warning: GPU memory ({gpu_memory:.1f}GB) may be insufficient")
            print(f"      Recommended: 80GB+ for GPT-OSS-20B")
        else:
            print(f"   ✅ GPU memory ({gpu_memory:.1f}GB) should be sufficient")
    else:
        print("   ❌ CUDA not available")
        return False
    
    # Test 2: Dataset loading
    print("\n2. Testing dataset loading...")
    try:
        dataset = load_dataset("Kaludi/Customer-Support-Responses", split="train")
        print(f"   ✅ Dataset loaded: {len(dataset)} samples")
        print(f"   Sample: {dataset[0]}")
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        return False
    
    # Test 3: Tokenizer loading
    print("\n3. Testing tokenizer loading...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
        print("   ✅ Tokenizer loaded successfully")
        print(f"   Vocabulary size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ❌ Tokenizer loading failed: {e}")
        return False
    
    # Test 4: LoRA configuration
    print("\n4. Testing LoRA configuration...")
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("   ✅ LoRA configuration created successfully")
    except Exception as e:
        print(f"   ❌ LoRA configuration failed: {e}")
        return False
    
    # Test 5: Basic PyTorch operations
    print("\n5. Testing basic PyTorch operations...")
    try:
        x = torch.randn(2, 3).cuda()
        y = torch.randn(2, 3).cuda()
        z = x + y
        print("   ✅ Basic PyTorch operations successful")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Basic PyTorch operations failed: {e}")
        return False
    
    # Test 6: Memory test
    print("\n6. Testing GPU memory allocation...")
    try:
        # Try to allocate a reasonable amount of memory
        test_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16).cuda()
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"   ✅ Memory allocation successful: {memory_allocated:.2f} GB allocated")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Memory allocation failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Your setup is ready for fine-tuning.")
    print("\n💡 Next steps:")
    print("   1. Ensure you have 80GB+ GPU memory")
    print("   2. Run: python finetune_gpt_oss_20b.py")
    print("   3. Monitor GPU memory usage during training")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    if not success:
        print("\n❌ Setup test failed. Please check the errors above.")
        exit(1)
