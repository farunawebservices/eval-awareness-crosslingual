#!/usr/bin/env python3
"""Assign train/val/test splits for H2 dataset."""
import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load prompts
    prompts = []
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    # Group by core_id
    by_core = {}
    for p in prompts:
        cid = p['core_id']
        if cid not in by_core:
            by_core[cid] = []
        by_core[cid].append(p)
    
    # Split cores: 35 train, 15 val, 20 test
    core_ids = list(by_core.keys())
    random.shuffle(core_ids)
    
    train_cores = set(core_ids[:35])
    val_cores = set(core_ids[35:50])
    test_cores = set(core_ids[50:])
    
    # Assign splits
    split_prompts = []
    for p in prompts:
        cid = p['core_id']
        if cid in train_cores:
            split = 'train'
        elif cid in val_cores:
            split = 'val'
        else:
            split = 'test'
        p['split'] = split
        split_prompts.append(p)
    
    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for p in split_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Stats
    train_count = sum(1 for p in split_prompts if p['split'] == 'train')
    val_count = sum(1 for p in split_prompts if p['split'] == 'val')
    test_count = sum(1 for p in split_prompts if p['split'] == 'test')
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

if __name__ == "__main__":
    main()
