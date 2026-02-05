import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import os
import os
st.write("Current Working Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())

# --- PAGE CONFIG ---
st.set_page_config(page_title="ArXiv Pulse", page_icon="🔬")
st.title("🔬 ArXiv Pulse: AI Research Assistant")
st.write("Search for papers and get AI-generated insights.")

# --- CACHING (Crucial for Speed) ---
# We use @st.cache_resource so we only load the heavy AI models ONCE, not on every click.
@st.cache_resource
def load_models():
    # 1. Load Data
    if not os.path.exists('arxiv_cleaned.csv') or not os.path.exists('arxiv_embeddings.pt'):
        # FIX: Return 6 Nones (You had 5 before)
        return None, None, None, None, None, None 
    
    df = pd.read_csv('arxiv_cleaned.csv')
    paper_embeddings = torch.load('arxiv_embeddings.pt', map_location=torch.device('cpu'))
    
    # 2. Load Models
    retriever_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    retriever_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    gen_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
    gen_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
    
    return df, paper_embeddings, retriever_tokenizer, retriever_model, gen_tokenizer, gen_model

# --- HELPER FUNCTIONS ---
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return torch.nn.functional.normalize(sum_embeddings / sum_mask, p=2, dim=1)

def generate_insight(query, title, abstract, tokenizer, model):
    prompt = f"Question: {query}\nPaper: {title}\nAbstract: {abstract}\nExplain relevance in one sentence."
    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100, num_beams=2, early_stopping=True) # Reduced beams for speed
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- MAIN APP LOGIC ---
df, paper_embeddings, ret_tok, ret_model, gen_tok, gen_model = load_models()

if df is None:
    st.error("Error: Could not find 'arxiv_cleaned.csv' or 'arxiv_embeddings.pt'. Please upload them.")
else:
    # Search Bar
    query = st.text_input("Enter a research topic:", "Machine Learning in Healthcare")

    if st.button("Analyze"):
        with st.spinner("Reading papers..."):
            # Retrieval
            query_vec = get_embedding(query, ret_tok, ret_model)
            scores = torch.nn.functional.cosine_similarity(query_vec, paper_embeddings)
            best_idx = torch.argmax(scores).item()
            best_paper = df.iloc[best_idx]
            
            # Display Result
            st.success(f"**Best Match:** {best_paper['title']}")
            st.write(f"**Abstract:** {best_paper['abstract'][:300]}...") # Show first 300 chars
            
            # Generation
            st.markdown("---")
            st.write("🤖 **AI Insight:**")
            insight = generate_insight(query, best_paper['title'], best_paper['abstract'], gen_tok, gen_model)
            st.info(insight)