#!/usr/bin/env python3
"""Render H2 dataset from cores and wrappers."""
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", required=True)
    parser.add_argument("--wrappers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # Import cores and wrappers
    import sys
    sys.path.insert(0, 'datasets')
    
    cores_module = __import__(args.cores.split('/')[-1].replace('.py', ''))
    wrappers_module = __import__(args.wrappers.split('/')[-1].replace('.py', ''))
    
    CORES = getattr(cores_module, 'CORES_YO')
    WRAPPERS = getattr(wrappers_module, 'WRAPPERS_YO')
    
    prompts = []
    for core_id, core in CORES.items():
        for fmt in ['eval_prose', 'deploy_prose', 'eval_mcq', 'deploy_mcq']:
            for fnum in ['F1', 'F2', 'F3', 'F4', 'F5']:
                wrapper_key = f"{fnum}_{fmt}"
                if wrapper_key not in WRAPPERS:
                    continue
                wrapper = WRAPPERS[wrapper_key]
                
                if 'mcq' in fmt:
                    prompt = wrapper.format(
                        ibeere=core['ibeere'],
                        A=core['esa']['A'],
                        B=core['esa']['B'],
                        C=core['esa']['C'],
                        D=core['esa']['D'],
                        leta=core['leta']
                    )
                else:
                    prompt = wrapper.format(
                        ibeere=core['ibeere'],
                        idahun=core['idahun']
                    )
                
                is_eval = 'eval' in fmt
                label = 1 if is_eval else 0
                
                prompts.append({
                    "core_id": core_id,
                    "domain": core['domain'],
                    "format": wrapper_key,
                    "prompt": prompt,
                    "label": label
                })
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    print(f"Rendered {len(prompts)} prompts to {args.output}")

if __name__ == "__main__":
    main()
