# if you have a folder with lots of .txt captions this lets you generate new prompts using markov chains.

import os
import random
from collections import defaultdict, Counter
from pathlib import Path
import re

def build_markov_chain(tokens, order=2):
    """Build Markov chain from token list with configurable order"""
    if order < 1:
        raise ValueError("Order must be at least 1")
    
    chain = defaultdict(Counter)
    
    for i in range(len(tokens) - order):
        # Use tuple of n tokens as key (where n = order)
        key = tuple(tokens[i:i + order])
        next_token = tokens[i + order]
        chain[key][next_token] += 1
    
    return chain

def generate_prompt(chain, length=15, order=2, start_with=None):
    """Generate new prompt using Markov chain with improved logic"""
    if not chain:
        return ""
    
    # Convert chain from Counter to probability-based selection
    prob_chain = {}
    for key, counter in chain.items():
        total = sum(counter.values())
        prob_chain[key] = [(token, count/total) for token, count in counter.items()]
    
    # Start with a random key or find one that starts with desired token
    if start_with:
        # Find keys that start with the desired token
        matching_keys = [key for key in chain.keys() 
                        if key[0].lower().startswith(start_with.lower())]
        current_key = random.choice(matching_keys) if matching_keys else random.choice(list(chain.keys()))
    else:
        current_key = random.choice(list(chain.keys()))
    
    result = list(current_key)
    
    for _ in range(length - order):
        if current_key in prob_chain:
            # Weighted random choice based on frequency
            choices, weights = zip(*[(token, weight) for token, weight in prob_chain[current_key]])
            next_token = random.choices(choices, weights=weights, k=1)[0]
            result.append(next_token)
            
            # Update the current key (slide the window)
            current_key = tuple(result[-order:])
        else:
            # If we hit a dead end, try to continue from a random point
            available_keys = [key for key in chain.keys() 
                            if key[0] == result[-1] or any(token in result for token in key)]
            if available_keys:
                current_key = random.choice(available_keys)
                # Add the remaining tokens from the new key (excluding the first which we already have)
                result.extend(current_key[1:])
            else:
                break
    
    # Clean up the result
    prompt = clean_prompt(result)
    return prompt

def clean_prompt(tokens):
    """Clean and format the prompt tokens"""
    # Remove duplicates while preserving order (to avoid repetition)
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    # Join and escape parentheses
    prompt = ", ".join(unique_tokens)
    prompt = prompt.replace("(", r"\(").replace(")", r"\)")
    
    # Remove excessive commas and spaces
    prompt = re.sub(r',\s*,', ', ', prompt)
    prompt = prompt.strip(', ')
    
    return prompt

def find_txt_files(folder):
    """Recursively find all .txt files"""
    return list(Path(folder).rglob("*.txt"))

def is_token_file(filepath):
    """Check if file contains comma-separated tokens"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return ',' in content and len(content) > 10  # Basic validation
    except:
        return False

def get_next_filename():
    """Find next available promptgen_x.txt filename"""
    i = 1
    while os.path.exists(f"promptgen_{i}.txt"):
        i += 1
    return f"promptgen_{i}.txt"

def analyze_tokens(tokens):
    """Analyze token statistics"""
    if not tokens:
        return {}
    
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    
    print(f"token analysis:")
    print(f"  total tokens: {total_tokens}")
    print(f"  unique tokens: {unique_tokens}")
    print(f"  most common tokens: {token_counts.most_common(10)}")
    
    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'token_counts': token_counts
    }

def main(folder_path=".", order=2, prompt_length=15, num_prompts=10):
    txt_files = find_txt_files(folder_path)
    all_tokens = []
    
    print(f"found {len(txt_files)} text files")
    
    for file in txt_files:
        if is_token_file(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # More robust token parsing
                    tokens = [t.strip() for t in re.split(r',|\n', content) if t.strip()]
                    all_tokens.extend(tokens)
                    # print(f"  loaded {len(tokens)} tokens from {file.name}")
            except Exception as e:
                print(f"  error reading {file}: {e}")
    
    if not all_tokens:
        print("no valid token files found")
        return
    
    # Analyze tokens
    analysis = analyze_tokens(all_tokens)
    
    if analysis['unique_tokens'] < order + 1:
        print(f"warning: not enough unique tokens for order {order}. using order 1.")
        order = 1
    
    chain = build_markov_chain(all_tokens, order)
    output_file = get_next_filename()
    
    print(f"generated markov chain with order {order}")
    print(f"chain has {len(chain)} unique states")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_prompts):
            prompt = generate_prompt(chain, length=prompt_length, order=order)
            f.write(f"{i+1}: {prompt}\n\n")
            print(f"generated prompt {i+1}")
    
    print(f"generated {num_prompts} prompts in {output_file}")

if __name__ == "__main__":
    # You can customize these parameters
    main(
        folder_path=".",
        order=2,           # Higher order = more context awareness
        prompt_length=15,  # Target prompt length
        num_prompts=10     # Number of prompts to generate
    )
