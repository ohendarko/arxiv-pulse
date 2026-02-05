import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os

# --- CONFIGUREATION --- 
DATA_FILE = 'arxiv_cleaned.csv'
EMBEDDING_FILE = 'arxiv_embeddings.pt'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 5

class RecommenderSystem:
  def __init__(self):
    print("Loading Knowledge Base...")

    # 1. Load the Data (Metadata)
    if not os.path.exists(DATA_FILE) or not os.path.exists(EMBEDDING_FILE):
      raise FileNotFoundError("Missing csv or pt file. Run previous steps.")
    
    self.df = pd.read_csv(DATA_FILE)

    # 2. Load the Embeddings (The "Brain")
    self.paper_embeddings = torch.load(EMBEDDING_FILE, map_location=torch.device('cpu'))

    # 3. Load Model for Query Processing
    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    self.model = AutoModel.from_pretrained(MODEL_NAME)

  def mean_pooling(self, model_output, attention_mask):
    # Same logic as before to vectorize the user's query
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
  
  def search(self, query):
    print(f"\nProcessing Query: '{query}'...")

    # A. Vectorize the Query
    inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
      outputs = self.model(**inputs)
    
    query_vec = self.mean_pooling(outputs, inputs['attention_mask'])
    query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=1)

    # B. Cosine Similarity (The "Magic")
    # computes similarity between Query (1 vector) and Papers (50 vectors)
    cosine_scores = torch.nn.functional.cosine_similarity(query_vec, self.paper_embeddings)

    # C. Ranking
    # Get top k scores and their indices
    top_results = torch.topk(cosine_scores, k=min(TOP_K, len(self.df)))

    print("\n--- Top Recommendations ---")
    for score, idx in zip(top_results.values, top_results.indices):
      idx = idx.item() # convert tensor to integer
      title = self.df.iloc[idx]['title']

      # Formatting the output
      print(f"[Score: {score:.4f} {title}]")
      # print(f"   -> Link: {self.df.iloc[idx]['url']}")

# --- INTERACTIVE TERMINAL ---
if __name__ == "__main__":
  rec_sys = RecommenderSystem()

  while True:
    user_query = input("\nEnter a search topic (or 'q' to quit)")
    if user_query.lower() == 'q':
      break

    rec_sys.search(user_query)