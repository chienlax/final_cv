import gzip
import json
import os
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
CATEGORY = "Beauty_and_Personal_Care"
DATA_DIR = "data"

# Input paths
REVIEW_PATH = os.path.join(DATA_DIR, f"{CATEGORY}.jsonl.gz")
META_PATH = os.path.join(DATA_DIR, f"meta_{CATEGORY}.jsonl.gz")

# Output paths
OUTPUT_REVIEW_PATH = os.path.join(DATA_DIR, f"filtered_{CATEGORY}.jsonl.gz")
OUTPUT_META_PATH = os.path.join(DATA_DIR, f"filtered_meta_{CATEGORY}.jsonl.gz")

def get_interaction_graph(file_path):
    """
    Reads only user_id and parent_asin to build the interaction graph efficiently.
    """
    users = []
    items = []
    
    print(f"Loading interaction graph from {file_path}...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading IDs"):
            try:
                data = json.loads(line)
                # Use parent_asin if available (groups variants), else asin
                item = data.get('parent_asin', data.get('asin'))
                user = data.get('user_id')
                
                if item and user:
                    users.append(user)
                    items.append(item)
            except json.JSONDecodeError:
                continue
                
    return pd.DataFrame({'user_id': users, 'parent_asin': items})

def filter_k_core(df, k=5):
    """
    Iteratively filters data until all users and items have at least k interactions.
    """
    print(f"Starting {k}-core filtering on {len(df)} interactions...")
    
    iteration = 0
    while True:
        iteration += 1
        start_count = len(df)
        
        # 1. Filter Users
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df['user_id'].isin(valid_users)]
        
        # 2. Filter Items
        item_counts = df['parent_asin'].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df['parent_asin'].isin(valid_items)]
        
        end_count = len(df)
        print(f"Iteration {iteration}: {start_count} -> {end_count} interactions")
        
        if start_count == end_count:
            print("Converged!")
            break
            
    return set(df['user_id'].unique()), set(df['parent_asin'].unique())

def filter_and_save(input_path, output_path, valid_ids, id_field, mode="Reviews"):
    """
    Streams the input file, keeps only records with valid IDs, and writes to output.
    """
    print(f"Filtering {mode} and saving to {output_path}...")
    
    kept = 0
    total = 0
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as fin, \
         gzip.open(output_path, 'wt', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc=f"Processing {mode}"):
            total += 1
            try:
                data = json.loads(line)
                # Check target ID (user_id/parent_asin for reviews, parent_asin for meta)
                target_id = data.get(id_field)
                
                # For reviews, we usually strictly need BOTH user and item to be valid
                # For metadata, we just need the item to be in the valid set
                if mode == "Reviews":
                    item_id = data.get('parent_asin', data.get('asin'))
                    user_id = data.get('user_id')
                    if item_id in valid_ids['items'] and user_id in valid_ids['users']:
                        fout.write(line)
                        kept += 1
                
                elif mode == "Metadata":
                    # For metadata, keep if item is in valid items list
                    if target_id in valid_ids['items']:
                        fout.write(line)
                        kept += 1
                        
            except json.JSONDecodeError:
                continue
                
    print(f"Finished {mode}. Kept {kept}/{total} records ({(kept/total)*100:.2f}%).")

if __name__ == "__main__":
    # Step 1: Load lightweight graph
    df = get_interaction_graph(REVIEW_PATH)
    
    # Step 2: Compute 5-core
    valid_users, valid_items = filter_k_core(df, k=5)
    print(f"Final Statistics: {len(valid_users)} Users, {len(valid_items)} Items.")
    
    # Group valid sets for easy passing
    valid_sets = {'users': valid_users, 'items': valid_items}
    
    # Step 3: Filter Reviews (Pass 2)
    filter_and_save(REVIEW_PATH, OUTPUT_REVIEW_PATH, valid_sets, 'parent_asin', mode="Reviews")
    
    # Step 4: Filter Metadata (Pass 3)
    filter_and_save(META_PATH, OUTPUT_META_PATH, valid_sets, 'parent_asin', mode="Metadata")
    
    print("\nAll done! Your clean data is in the 'data/' folder.")