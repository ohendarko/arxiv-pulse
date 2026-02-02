import arxiv
import pandas as pd
import re
import time
import json

# null = None

class ArxivIngestorLocal:
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

  def load_data(self, max_records=1000):
    """
    Reads the file line-by-line to handle the JSON format safely.
    """
    papers_data = []

    print(f"Reading from {self.file_path}...")

    try:
      with open(self.file_path, 'r') as f:
        for i, line in enumerate(f):
          if i >= max_records:
            break

          try:
            # json.loads automatically converts 'null' to None
            paper = json.loads(line)

            # Extract only what we need
            papers_data.append({
              "id": paper.get("id"),
              "title": paper.get("title"),
              "abstract": self.clean_text(paper.get("abstract")),
              "categories": paper.get("categories"),
              "doi": paper.get("doi")
            })
          except json.JSONDecodeError:
            print(f"Skipping malformed line {i}")
            continue
    except FileNotFoundError:
      print(f"Error: The file '{self.file_path}' ws not found.")
      # return pd.DataFrame()
    
    return pd.DataFrame(papers_data)
  
# --- Execution ---
if __name__ == "__main__":
  # Make sure your json file is in the same folder as this script
    # If using the Kaggle dataset, the file is usually 'arxiv-metadata-oai-snapshot.json'
    ingestor = ArxivIngestorLocal("arxiv-sample.json")
    df = ingestor.load_data(max_records=50) # Just load 50 for testing
    
    # Check if the cleaning worked
    if not df.empty:
        print("\n--- Success! Loaded Data ---")
        print(df[['title', 'abstract']].head(2))
        
        # Save to CSV for Day 2
        df.to_csv("arxiv_cleaned.csv", index=False)
        print("\nSaved to arxiv_cleaned.csv")
    else:
        print("No data loaded. Check your file name.")

