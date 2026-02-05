import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import os

# --- CONFIGURATION ---
# --- CONFIGURATION ---
DATA_FILE = 'arxiv_cleaned.csv'
EMBEDDING_FILE = 'arxiv_embeddings.pt'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GENERATION_MODEL = 'google/flan-t5-small' # A small, fast LLM for CPU usage

class ArxivAgent:
  def __init__(self):
    print("--- Initializing AI Agent ---")

    # 1. Load Data & Embeddings
    self.df = pd.read_csv(DATA_FILE)
    self.paper_embeddings = torch.load(EMBEDDING_FILE, map_location=torch.device('cpu'))

    # 2. Load Embedding Model (The "Retriever")
    print("Loading Retriever...")
    self.retriever_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    self.retriever_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

    # 3. Load Generative Model (The "Gneerator")
    print("Loading Generator (LLM)...")
    self.gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)

  def get_embedding(self, text):
    inputs = self.retriever_tokenizer(text, padding=True, return_tensors='pt')
    with torch.no_grad():
      outputs = self.retriever_model(**inputs)

    # Mean Pooling
    token_embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)
  
  def generate_insight(self, query, paper_title, paper_abstract):
      # The Prompt Engineering part
      prompt = (
          f"User Question: {query}\n"
          f"Paper Title: {paper_title}\n"
          f"Abstract: {paper_abstract}\n\n"
          f"Task: Explain in one sentence why this paper is relevant to the question."
      )

      inputs = self.gen_tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
      outputs = self.gen_model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)

      return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

  def run(self, user_query):
      print(f"\nThinking about: '{user_query}'...")

      # A. Retrieval
      query_vec = self.get_embedding(user_query)
      scores = torch.nn.functional.cosine_similarity(query_vec, self.paper_embeddings)
      best_idx = torch.argmax(scores).item()

      best_paper = self.df.iloc[best_idx]
      print(f"--> Found Best Match: {best_paper['title']}")

      # B. Generation (The AI Agent speaks)
      print("--> Reading abstract and generating insight...")
      insight = self.generate_insight(user_query, best_paper['title'], best_paper['clean_abstract'])

      print(f"\n[AI AGENT SAYS]: {insight}")

# --- EXECUTION ---
if __name__ == "__main__":
   agent = ArxivAgent()

   while True:
      q = input("\nAsk a research question (or 'q' to quit): ")
      if q.lower() == 'q': break
      agent.run(q)