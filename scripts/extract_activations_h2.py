#!/usr/bin/env python3
"""Extract activations for H2 dataset."""
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", type=int, nargs='+', default=[12, 16, 20, 24])
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load prompts
    prompts = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    # Extract activations
    activations = {}
    manifest = {"prompts": [], "layers": args.layers}
    
    for i, p in enumerate(prompts):
        prompt_text = p['prompt']
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        act_dict = {}
        for layer in args.layers:
            # Get last token activation
            layer_act = hidden_states[layer][0, -1, :].cpu()
            act_dict[f"layer_{layer}"] = layer_act.tolist()
        
        activations[i] = act_dict
        manifest["prompts"].append({
            "idx": i,
            "core_id": p['core_id'],
            "split": p['split'],
            "label": p['label'],
            "format": p['format']
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(prompts)} prompts")
    
    # Save
    torch.save(activations, args.output)
    with open(args.manifest, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved activations to {args.output}")
    print(f"Saved manifest to {args.manifest}")

if __name__ == "__main__":
    main()
