import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings
embeddings = np.load("embeddings.npy")  # Ensure this file exists

# Load document content
documents = {}
with open("documents.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(": ", 1)  # Split into ID and content
        if len(parts) == 2:
            doc_id, content = parts
            documents[doc_id] = content

# ✅ Function to retrieve the Top-K most similar documents
def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    document_ids = list(documents.keys())
    return [(document_ids[i], similarities[i]) for i in top_k_indices]

# ✅ Function to convert user query into an embedding
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])  # Placeholder - random embedding

# ✅ Streamlit UI Styling
st.set_page_config(page_title="Information Retrieval", page_icon="🔍", layout="wide")
st.markdown(
    """
    <style>
        body { font-family: 'Arial', sans-serif; }
        .stApp { background-color: #1E1E1E; color: white; }
        .stTextInput input { border-radius: 8px; }
        .result-card {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            background-color: #2E2E2E;
            box-shadow: 2px 2px 5px rgba(255, 255, 255, 0.1);
        }
        .doc-id { font-weight: bold; color: #FFD700; }
        .score { color: #66FF66; }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar for user input
st.sidebar.title("🔍 Search Engine")
st.sidebar.write("Enter a query to find relevant documents.")
query = st.sidebar.text_input("Search Query:")

if st.sidebar.button("Search"):
    if not query.strip():
        st.sidebar.warning("⚠️ Please enter a valid query!")
    else:
        query_embedding = get_query_embedding(query)
        results = retrieve_top_k(query_embedding, embeddings)
        
        st.sidebar.success(f"✅ Found {len(results)} relevant documents!")
        
        # ✅ Display the results with improved UI
        st.write("## 🏆 Top 10 Relevant Documents")
        
        for doc_id, score in results:
            doc_content = documents.get(doc_id, "⚠️ Content not found")
            with st.container():
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'<span class="doc-id">📄 {doc_id}</span> <span class="score">(Score: {score:.4f})</span>', unsafe_allow_html=True)
                st.write(f"📝 {doc_content[:500]}...")
                st.markdown('</div>', unsafe_allow_html=True)
                st.write("---")
