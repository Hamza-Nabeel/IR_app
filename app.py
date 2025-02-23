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
            documents[doc_id] = content  # Store the content


# ✅ Function to retrieve the Top-K most similar documents
def retrieve_top_k(query_embedding, embeddings, k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]  # Get top-k most similar

    # Ensure indices do not go out of bounds
    document_ids = list(documents.keys())  # Get document IDs
    return [(document_ids[i], similarities[i]) for i in top_k_indices]


# ✅ Function to convert user query into an embedding
# (Replace this with an actual NLP model for embeddings)
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1])  # Placeholder - random embedding


# ✅ Streamlit UI
st.title("🔍 Information Retrieval System")
st.subheader("Search Reuters News Articles Using Word Embeddings")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query!")
    else:
        query_embedding = get_query_embedding(query)
        st.write(f"Documents loaded: {len(documents)}")
        st.write(f"Embeddings shape: {embeddings.shape}")

        results = retrieve_top_k(query_embedding, embeddings)

        # ✅ Display the results with actual document content
        st.write("### 🏆 Top 10 Relevant Documents:")
        for doc_id, score in results:
            doc_content = documents.get(doc_id, "⚠️ Content not found")
            st.write(f"📄 **{doc_id}** (Score: {score:.4f})")
            st.write(f"📝 {doc_content[:500]}...")  # Show first 500 characters
            st.write("---")
