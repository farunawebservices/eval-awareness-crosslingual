#!/usr/bin/env python3
"""Fast activation extraction with batching."""
import json
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        p = self.prompts[idx]
        enc = self.tokenizer(
            p['prompt'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'idx': idx,
            'prompt': p
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", type=int, nargs='+', default=[12, 16, 20, 24])
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    print(f"Loading model {args.model} in float16...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Load prompts
    prompts = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    dataset = PromptDataset(prompts, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    activations = {}
    manifest = {"prompts": [], "layers": args.layers}
    
    print(f"Processing {len(prompts)} prompts in batches of {args.batch_size}...")
    
    for batch in loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        indices = batch['idx'].tolist()
        batch_prompts = batch['prompt']
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states
        
        for i, (idx, p) in enumerate(zip(indices, batch_prompts)):
            act_dict = {}
            for layer in args.layers:
                # Get last non-padded token
                mask = attention_mask[i]
                last_idx = mask.sum() - 1
                layer_act = hidden_states[layer][i, last_idx, :].cpu().float()
                act_dict[f"layer_{layer}"] = layer_act.numpy()
            
            activations[idx] = act_dict
            manifest["prompts"].append({
                "idx": idx,
                "core_id": p['core_id'],
                "split": p['split'],
                "label": p['label'],
                "format": p['format']
            })
        
        if (len(activations) % 100) == 0:
            print(f"Processed {len(activations)}/{len(prompts)}")
    
    # Save
    torch.save(activations, args.output)
    with open(args.manifest, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved {len(activations)} activations to {args.output}")

if __name__ == "__main__":
    main()
