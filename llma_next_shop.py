import os
import torch
from transformers import LlamaTokenizerFast, LlamaForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Explicitly disable MPS backend
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

device = torch.device("cpu")
torch.set_default_tensor_type(torch.FloatTensor)

# Step 1: Define special tokens for shops, categories, and times
shop_tokens = ["<shop_A>", "<shop_B>", "<shop_C>", "<shop_D>"]
cat_tokens = ["<cat_food>", "<cat_clothes>", "<cat_electronics>"]
time_tokens = ["<t1>", "<t2>", "<t3>", "<t4>"]

# Step 2: Load LLaMA tokenizer and add tokens
# Note: You can use "meta-llama/Llama-2-7b-hf" if you have access, or a smaller model like "JackFram/llama-160m"
model_name = "JackFram/llama-160m"  # Using a smaller LLaMA model for demonstration
tokenizer = LlamaTokenizerFast.from_pretrained(model_name)

# LLaMA doesn't have a pad token by default, so we add one
tokenizer.add_special_tokens({
    "additional_special_tokens": shop_tokens + cat_tokens + time_tokens,
    "pad_token": "<pad>",
    "eos_token": "</s>",  # LLaMA's default EOS
    "bos_token": "<s>",   # LLaMA's default BOS
})

# Step 3: Load LLaMA model and resize embeddings
model = LlamaForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Step 4: Create training data with clearer patterns
# Note: LLaMA typically uses BOS and EOS tokens
examples = [
    # Pattern: BOS -> t1 -> shop -> category -> t2 -> different shop -> different category -> t3 -> third shop -> category -> EOS
    "<s> <t1> <shop_A> <cat_food> <t2> <shop_B> <cat_clothes> <t3> <shop_C> <cat_electronics> </s>",
    "<s> <t1> <shop_B> <cat_clothes> <t2> <shop_C> <cat_electronics> <t3> <shop_A> <cat_food> </s>",
    "<s> <t1> <shop_C> <cat_electronics> <t2> <shop_A> <cat_food> <t3> <shop_B> <cat_clothes> </s>",
    "<s> <t1> <shop_D> <cat_food> <t2> <shop_A> <cat_electronics> <t3> <shop_B> <cat_food> </s>",
    "<s> <t1> <shop_A> <cat_electronics> <t2> <shop_D> <cat_food> <t3> <shop_C> <cat_clothes> </s>",
    "<s> <t1> <shop_B> <cat_food> <t2> <shop_C> <cat_clothes> <t3> <shop_D> <cat_electronics> </s>",
    "<s> <t1> <shop_C> <cat_clothes> <t2> <shop_D> <cat_electronics> <t3> <shop_A> <cat_food> </s>",
    "<s> <t1> <shop_D> <cat_electronics> <t2> <shop_B> <cat_food> <t3> <shop_C> <cat_clothes> </s>",
    # Add some 2-step sequences
    "<s> <t1> <shop_A> <cat_food> <t2> <shop_B> <cat_clothes> </s>",
    "<s> <t1> <shop_C> <cat_electronics> <t2> <shop_D> <cat_food> </s>",
    "<s> <t1> <shop_B> <cat_clothes> <t2> <shop_A> <cat_electronics> </s>",
    # Add some 4-step sequences
    "<s> <t1> <shop_A> <cat_food> <t2> <shop_B> <cat_clothes> <t3> <shop_C> <cat_electronics> <t4> <shop_D> <cat_food> </s>",
    "<s> <t1> <shop_D> <cat_electronics> <t2> <shop_A> <cat_food> <t3> <shop_B> <cat_clothes> <t4> <shop_C> <cat_electronics> </s>",
]

class ShopDataset(Dataset):
    def __init__(self, texts):
        # Use longer max_length to accommodate full sequences
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=40, return_tensors="pt")
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {k: v[idx].to(device) for k, v in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        # Set labels for padding tokens to -100 (ignore in loss calculation)
        item["labels"][item["labels"] == tokenizer.pad_token_id] = -100
        return item

train_dataset = ShopDataset(examples)

# Step 5: Training arguments optimized for LLaMA
training_args = TrainingArguments(
    output_dir="./llama_results",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    logging_steps=5,
    save_strategy="no",
    use_cpu=True,
    dataloader_pin_memory=False,
    no_cuda=True,
    learning_rate=5e-4,
    warmup_steps=10,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    fp16=False,  # Disable fp16 for CPU
    gradient_checkpointing=False,  # Disable for small model
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()
print("Training completed!")

def generate_next_sequence(prompt, max_new_tokens=10, temperature=0.7, do_sample=True, top_k=50, top_p=0.9, show_logits=False):
    """Generate continuation for a given prompt"""
    # Add BOS token if not present
    if not prompt.startswith("<s>"):
        prompt = "<s> " + prompt
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if show_logits:
            return generate_with_logits(prompt, max_new_tokens, temperature, do_sample, top_k, top_p)
        else:
            # Standard generation
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
            )
            
            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            # Remove BOS token from display if it was added
            display_prompt = prompt.replace("<s> ", "") if prompt.startswith("<s> ") else prompt
            return display_prompt + generated_text

def generate_with_logits(prompt, max_new_tokens=10, temperature=0.7, do_sample=True, top_k=50, top_p=0.9):
    """Generate text step by step while showing logits for each token"""
    current_text = prompt
    
    print(f"=== STEP-BY-STEP GENERATION WITH LOGITS ===")
    print(f"Initial prompt: {prompt}")
    print()
    
    for step in range(max_new_tokens):
        inputs = tokenizer(current_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            
            probabilities = torch.softmax(logits, dim=-1)
            
            # Show top predictions
            top_probs, top_indices = torch.topk(probabilities, 10)
            print(f"Step {step + 1} - Top 10 predictions:")
            print("Token | Logit | Probability")
            print("-" * 35)
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = tokenizer.decode([idx])
                logit_val = logits[idx].item()
                
                # Handle special tokens and whitespace for display
                if token.startswith('<') and token.endswith('>'):
                    display_token = token
                elif token.strip() == '':
                    display_token = f"'{token}'" if len(token) == 1 else f"SPACE({len(token)})"
                else:
                    display_token = f"'{token}'"
                
                print(f"{display_token:<15} | {logit_val:6.2f} | {prob.item():.4f}")
            
            # Sample next token
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    filtered_logits = torch.full_like(logits, float('-inf'))
                    filtered_logits.scatter_(0, top_k_indices, top_k_logits)
                    logits = filtered_logits
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_token = tokenizer.decode(next_token_id)
            selected_logit = logits[next_token_id].item()
            selected_prob = probabilities[next_token_id].item()
            
            display_next = next_token if next_token.startswith('<') and next_token.endswith('>') else f"'{next_token}'"
            print(f">>> SELECTED: {display_next} (logit: {selected_logit:.2f}, prob: {selected_prob:.4f})")
            print()
            
            current_text += next_token
            
            if next_token_id.item() == tokenizer.eos_token_id:
                print("EOS token generated, stopping.")
                break
    
    print(f"Final generated text: {current_text}")
    return current_text

print("\n=== INFERENCE EXAMPLES ===\n")

# Example 1: Predict next in sequence
print("1. Predicting next shop in sequence:")
prompt1 = "<t1> <shop_A> <cat_food> <t2> <shop_B> <cat_clothes> <t3>"
result1 = generate_next_sequence(prompt1, max_new_tokens=6, do_sample=False)
print(f"Input:  {prompt1}")
print(f"Output: {result1}")
print()

# Example 2: Complete 2-step sequence
print("2. Complete 2-step sequence:")
prompt2 = "<t1> <shop_C> <cat_electronics> <t2>"
result2 = generate_next_sequence(prompt2, max_new_tokens=8, do_sample=False)
print(f"Input:  {prompt2}")
print(f"Output: {result2}")
print()

# Example 3: Start new sequence
print("3. Start new sequence:")
prompt3 = "<t1>"
result3 = generate_next_sequence(prompt3, max_new_tokens=12, do_sample=False)
print(f"Input:  {prompt3}")
print(f"Output: {result3}")
print()

# Example 4: Multiple sampling attempts
print("4. Multiple generations with sampling:")
prompt4 = "<t1> <shop_A> <cat_food> <t2>"
for i in range(3):
    result = generate_next_sequence(prompt4, max_new_tokens=8, temperature=0.8)
    print(f"Attempt {i+1}: {result}")
print()

# Example 5: Test model's pattern recognition
print("5. Pattern recognition test:")
test_cases = [
    "<t1> <shop_A> <cat_food>",
    "<t2> <shop_B> <cat_clothes>", 
    "<t1> <shop_D> <cat_electronics> <t2>",
    "<t1> <shop_C> <cat_clothes> <t2> <shop_A>",
]

for prompt in test_cases:
    result = generate_next_sequence(prompt, max_new_tokens=8, do_sample=False)
    print(f"'{prompt}' -> continuation: '{result[len(prompt):]}'")
print()

# Example 6: Check if model learned the vocabulary properly
print("6. Vocabulary check - most likely tokens after specific prompts:")
test_prompts = [
    "<t1>",
    "<t1> <shop_A>",
    "<t1> <shop_A> <cat_food> <t2>",
]

for prompt in test_prompts:
    # Add BOS token if not present
    if not prompt.startswith("<s>"):
        prompt_with_bos = "<s> " + prompt
    else:
        prompt_with_bos = prompt
        
    inputs = tokenizer(prompt_with_bos, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        print(f"After '{prompt}', most likely next tokens:")
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode([idx])
            if token.strip():
                print(f"  '{token}': {prob.item():.4f}")
        print()

print("Done!")