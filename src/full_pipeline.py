"""
Comprehensive pipeline for Multimodal Recommendation System (Beauty Dataset).

This script handles data ingestion, preprocessing, dataset creation,
model definition, and training/evaluation, adhering to PEP8 standards.
"""

import gzip
import json
import logging
import os
import random
import hashlib
from typing import Dict, List, Tuple, Generator, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration parameters for the pipeline."""
    DATA_DIR: str = "data"
    REVIEW_FILE: str = "Beauty_and_Personal_Care.jsonl.gz"
    # META_FILE: str = "meta_Beauty_and_Personal_Care.jsonl.gz" # Unused in this text-focused pipeline
    SUBSET_RATIO: int = 10  # 1/10th of data via Modulo
    MIN_INTERACTIONS: int = 5  # K-core filter
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 32
    EPOCHS: int = 3
    LEARNING_RATE_BERT: float = 1e-5
    LEARNING_RATE_EMB: float = 1e-3
    EMBEDDING_DIM: int = 64
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Sets reproducibility seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Phase 1 & 2: Data Ingestion, Exploration, and Preprocessing
# ------------------------------------------------------------------------------

def stream_and_subset_data(file_path: str, subset_mod: int = 10) -> List[Dict[str, Any]]:
    """
    Streams data from a GZIP JSONL file and subsets it using hashing.
    
    Args:
        file_path: Path to the .jsonl.gz file.
        subset_mod: Modulo divisor for hashing (e.g., 10 for 10% subset).
        
    Returns:
        A list of dictionary records belonging to the subset.
    """
    data_subset = []
    logger.info(f"Streaming and subsetting data from {file_path}...")

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 5:  # Inspect Structure
                    logger.debug(f"Row {i} preview: {line.strip()[:200]}")
                
                try:
                    record = json.loads(line)
                    # Key checks
                    user_id = record.get('reviewerID')
                    item_id = record.get('parent_asin')  # Preferred over 'asin'
                    text = record.get('text', '')
                    rating = record.get('rating')

                    if not (user_id and item_id and rating is not None):
                        continue
                    
                    if not text.strip():
                        continue

                    # Scalable Subsetting via Hashing
                    # We hash user_id to keep all interactions for selected users (user-centric sampling)
                    # or hash interaction for random sampling. Plan requests User preservation?
                    # "If you select a user, you get all their interactions" -> Hash user_id.
                    hash_object = hashlib.md5(user_id.encode())
                    hash_int = int(hash_object.hexdigest(), 16)
                    
                    if hash_int % subset_mod == 0:
                        data_subset.append({
                            'user_id': user_id,
                            'item_id': item_id,
                            'text': text,
                            'rating': float(rating)
                        })

                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    logger.info(f"Subset complete. Collected {len(data_subset)} records.")
    return data_subset


def apply_k_core_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Iteratively filters data to ensure all users and items have at least k interactions.
    """
    logger.info(f"Applying {k}-core filter...")
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()

        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index

        # Filter
        new_df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]

        if len(new_df) == len(df):
            break
        df = new_df
    
    logger.info(f"K-core filter complete. Remaining records: {len(df)}")
    return df.reset_index(drop=True)


def create_id_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], pd.DataFrame]:
    """
    Creates integer mappings for User IDs and Item IDs.
    """
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: i for i, i in enumerate(item_ids)}

    df['user_idx'] = df['user_id'].map(user2idx)
    df['item_idx'] = df['item_id'].map(item2idx)

    logger.info(f"Mapped {len(user_ids)} users and {len(item_ids)} items.")
    return user2idx, item2idx, df


# ------------------------------------------------------------------------------
# Phase 3: Dataset Implementation
# ------------------------------------------------------------------------------

class BeautyDataset(Dataset):
    """
    Custom PyTorch Dataset for Multimodal Recommendation.
    Handles tokenization on-the-fly to save memory.
    """
    def __init__(self, 
                 users: np.ndarray, 
                 items: np.ndarray, 
                 texts: np.ndarray, 
                 ratings: np.ndarray, 
                 tokenizer: Any, 
                 max_len: int):
        self.users = users
        self.items = items
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        # Tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'user_idx': torch.tensor(self.users[idx], dtype=torch.long),
            'item_idx': torch.tensor(self.items[idx], dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


# ------------------------------------------------------------------------------
# Phase 4: Model Architecture
# ------------------------------------------------------------------------------

class MultimodalRecommender(nn.Module):
    """
    Hybrid model combining Collaborative Filtering (Embeddings) and Content (Text).
    Structure:
      - User Embedding
      - Item Embedding
      - Text Encoder (DistilBERT)
      - Concatenation -> MLP -> Rating
    """
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 embedding_dim: int, 
                 dropout: float = 0.2):
        super(MultimodalRecommender, self).__init__()
        
        # Collaborative Filtering Branch
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Content Branch (Text)
        # Using a smaller config or pre-trained model. 
        # Note: Fine-tuning BERT is expensive; often we freeze layers.
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_hidden_dim = self.bert.config.hidden_size
        
        # Fusion Layer
        # Input: UserEmb (dim) + ItemEmb (dim) + TextEmb (768)
        combined_dim = embedding_dim + embedding_dim + self.bert_hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Regression output
        )
        
    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, 
                input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. Graph/ID Embeddings
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        
        # 2. Text Embeddings
        # Extract CLS token output (index 0)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_output.last_hidden_state[:, 0, :]
        
        # 3. Fusion
        concat_vector = torch.cat([u_emb, i_emb, text_emb], dim=1)
        
        # 4. Prediction
        output = self.mlp(concat_vector)
        return output.squeeze()


# ------------------------------------------------------------------------------
# Phase 5: Training & Evaluation
# ------------------------------------------------------------------------------

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: str) -> float:
    
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        user_idx = batch['user_idx'].to(device)
        item_idx = batch['item_idx'].to(device)
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        rating = batch['rating'].to(device)
        
        optimizer.zero_grad()
        
        prediction = model(user_idx, item_idx, input_ids, mask)
        loss = criterion(prediction, rating)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: str) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            rating = batch['rating'].to(device)
            
            prediction = model(user_idx, item_idx, input_ids, mask)
            loss = criterion(prediction, rating)
            
            total_loss += loss.item()
            all_preds.extend(prediction.cpu().numpy())
            all_targets.extend(rating.cpu().numpy())
            
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, rmse


# ------------------------------------------------------------------------------
# Phase 6: Main Execution
# ------------------------------------------------------------------------------

def main():
    set_seed(Config.SEED)
    file_path = os.path.join(Config.DATA_DIR, Config.REVIEW_FILE)
    
    # --- Step 1 & 2: Data Ingestion & Preprocessing ---
    # Check if data exists
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}. Please download/place data.")
        return

    # Stream & Subset
    raw_data = stream_and_subset_data(file_path, subset_mod=Config.SUBSET_RATIO)
    df = pd.DataFrame(raw_data)
    
    # Filter
    df = apply_k_core_filter(df, k=Config.MIN_INTERACTIONS)
    
    # Map IDs
    user2idx, item2idx, df = create_id_mappings(df)
    
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    # Split Data (Time-based split preferred, but random stratified here for simplicity)
    train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=Config.SEED)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=Config.SEED)
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # --- Step 3: Dataset & Dataloader ---
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def create_loader(dataframe: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = BeautyDataset(
            users=dataframe['user_idx'].values,
            items=dataframe['item_idx'].values,
            texts=dataframe['text'].values,
            ratings=dataframe['rating'].values,
            tokenizer=tokenizer,
            max_len=Config.MAX_LENGTH
        )
        return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=shuffle, num_workers=2)

    train_loader = create_loader(train_df, shuffle=True)
    val_loader = create_loader(val_df, shuffle=False)
    test_loader = create_loader(test_df, shuffle=False)
    
    # --- Step 4: Model Initialization ---
    model = MultimodalRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(Config.DEVICE)
    
    # Optimizer: Separate LR for BERT and Embeddings/MLP
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': Config.LEARNING_RATE_BERT},
        {'params': model.user_embedding.parameters(), 'lr': Config.LEARNING_RATE_EMB},
        {'params': model.item_embedding.parameters(), 'lr': Config.LEARNING_RATE_EMB},
        {'params': model.mlp.parameters(), 'lr': Config.LEARNING_RATE_EMB}
    ])
    
    criterion = nn.MSELoss()
    
    # --- Step 5: Training Loop ---
    logger.info("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f}"
        )
        
    # Final Test
    test_loss, test_rmse = evaluate(model, test_loader, criterion, Config.DEVICE)
    logger.info(f"Final Test RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()