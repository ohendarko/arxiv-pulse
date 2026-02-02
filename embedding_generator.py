import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'arxiv_cleaned.csv'
OUTPUT_FILE = 'arxiv_embeddigs.pt'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 32

# 1. Setup Device (Crucial for the "CUDA" requirement)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

class EmbeddingEngine:
    def __init__(self, model_name):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
      
    def mean_pooling(self, model_output, attention_mask):
        """
        Manually pools token embeddings into a single sentence vector.
        This demonstrates understanding of the underlying matrix algebra.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings and divide by the number of valid tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def generate(self, text_list):
        all_embeddings = []
        total = len(text_list)

        print(f"Starting inference on {total} papers...")

        # Batch processing to prevent Memory Errors
        for i in range(0, total, BATCH_SIZE):
            batch = text_list[i: i + BATCH_SIZE]

            # Tokenize
            encoded_input = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            # Inference (No Gradients needed = faster, less RAM)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Pool and Normalize
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            # Move to CPU to save RAM, then store
            all_embeddings.append(sentence_embeddings.cpu())

            if i % 100 == 0:
                print(f"Processed {i}/{total}...")
        
        # Concatenate all batches into one large tensor
        return torch.cat(all_embeddings, dim=0)
    

# --- EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the previous script first.")
    else:
        # Load Data
        df = pd.read_csv(INPUT_FILE)
        
        # Ensure no empty values crash the model
        df['abstract'] = df['abstract'].fillna("No abstract available")
        abstracts = df['abstract'].tolist()
        
        # Generate
        engine = EmbeddingEngine(MODEL_NAME)
        embeddings = engine.generate(abstracts)
        
        # Save Tensor
        torch.save(embeddings, OUTPUT_FILE)
        print(f"\nSUCCESS: Saved embeddings tensor to {OUTPUT_FILE}")
        print(f"Tensor Shape: {embeddings.shape}")