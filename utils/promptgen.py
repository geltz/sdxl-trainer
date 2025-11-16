# generate new prompts using markov chains
# uses .txt files that have comma-separated captions

import os
import random
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def build_chain_streaming(files, order=2):
    """Build chain directly from files"""
    chain = defaultdict(Counter)
    window = []
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                tokens = [t.strip() for t in f.read().replace('\n', ',').split(',') if t.strip()]
                for token in tokens:
                    window.append(token)
                    if len(window) > order:
                        chain[tuple(window[-order-1:-1])][window[-1]] += 1
        except:
            pass
    return chain

def merge_chains(chain1, chain2):
    """Merge two chains"""
    result = defaultdict(Counter, chain1)
    for key, counter in chain2.items():
        result[key].update(counter)
    return result

def generate_prompt(chain, length=15, order=2):
    """Generate prompt from chain"""
    if not chain:
        return ""
    
    current_key = random.choice(list(chain.keys()))
    result = list(current_key)
    
    for _ in range(length - order):
        if current_key not in chain:
            break
        counter = chain[current_key]
        choices, weights = zip(*counter.items())
        result.append(random.choices(choices, weights=weights)[0])
        current_key = tuple(result[-order:])
    
    seen = set()
    unique = [t for t in result if not (t in seen or seen.add(t))]
    return ", ".join(unique).replace("(", r"\(").replace(")", r"\)")

def get_next_filename():
    """Find next available promptgen_x.txt"""
    i = 1
    while os.path.exists(f"promptgen_{i}.txt"):
        i += 1
    return f"promptgen_{i}.txt"

def build_chain_wrapper(args):
    """Wrapper for parallel execution"""
    batch, order = args
    return build_chain_streaming(batch, order)

def main(folder_path=".", order=2, prompt_length=15, num_prompts=10, max_workers=None):
    txt_files = list(Path(folder_path).rglob("*.txt"))
    print(f"found {len(txt_files)} text files")
    
    num_workers = max_workers or os.cpu_count()
    batch_size = max(1, len(txt_files) // num_workers)
    file_batches = [txt_files[i:i + batch_size] for i in range(0, len(txt_files), batch_size)]
    
    print(f"processing with {num_workers} workers")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chains = list(executor.map(build_chain_wrapper, [(b, order) for b in file_batches]))
    
    print("merging chains...")
    chain = chains[0]
    for c in chains[1:]:
        chain = merge_chains(chain, c)
    
    print(f"chain has {len(chain)} states, generating prompts...")
    
    output_file = get_next_filename()
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_prompts):
            prompt = generate_prompt(chain, length=prompt_length, order=order)
            f.write(f"{i+1}: {prompt}\n\n")
    
    print(f"generated {num_prompts} prompts in {output_file}")

if __name__ == "__main__":
    main(folder_path=".", order=2, prompt_length=15, num_prompts=10, max_workers=None)