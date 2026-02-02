import arxiv
import pandas as pd
import re
import time
import json

class ArxivIngestor:
  def __init__(self):
    self.client = arxiv.Client()

  def clean_text(self, text):
    """
    Simulates 'cleaning messy data'. 
    Removes LaTeX math, newlines, and extra spaces.
    """
    if not text:
        return ""
    
    # 1. Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # 2. Remove LaTeX math patterns (anything between $...$)
    # This is crucial for NLP - math symbols are noise for semantic understanding
    text = re.sub(r'\$.*?\$', '', text)
    
    # 3. Remove multiple spaces resulting from the above
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
  
  def fetch_papers(self, query="cat:cs.AI", max_results=100):
    """
    Fetches papers from ArXiv. 
    'cat:cs.AI' fetches Artificial Intelligence papers.
    """
    print(f"Fetching {max_results} papers for query: {query}...")

    search = arxiv.Search(
       query=query,
       max_results=max_results,
       sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers_data = []

    try:
        results = self.client.results(search)
        for r in results:
            # I only want the data we can actually use
            papers_data.append({
               "title": r.title,
               "published": r.published.strftime("%Y-%m-%d"),
               "raw_abstract": r.summary,
               "clean_abstract": self.clean_text(r.summary),
               "url": r.pdf_url,
               "category": r.primary_category
            })
    except Exception as e:
        print(f"Error fetching data: {e}")
    
    return pd.DataFrame(papers_data)
  
# --- Execution ---
if __name__ == "__main__":
    ingestor = ArxivIngestor()
    # Let's fetching papers on Machine Learning and Health Informatics
    # eg: combining cs.LG (Learning) and q-bio (Quantitative Biology)
    df = ingestor.fetch_papers(query="cat:cs.LG OR cat:cs.AI", max_results=50)

    # Inspect the "Messy" vs "Clean" data
    print("\n--- Data Sample ---")
    print(df[['raw_abstract', 'clean_abstract']].head(1))

    df.to_csv("arxiv_cleaned.csv", index=False)
    print(f"\nSaved {len(df)} papers to arxiv_cleaned.csv")
