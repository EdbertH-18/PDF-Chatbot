import streamlit as st
from utils import load_pdf, embed_chunks, retrieve_relevant_chunks, generate_groq_answer
from sentence_transformers import SentenceTransformer
import torch

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Ask Your PDF")
st.markdown("This chatbot reads your PDF and answers questions based on its content.")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --- Upload PDF
doc = st.file_uploader("Upload a PDF", type="pdf")

if doc:
    st.success("PDF uploaded successfully.")
    chunks = load_pdf(doc)
    chunk_embeddings = embed_chunks(chunks, embedder)

    question = st.text_input("Ask a question based on this PDF:")
    if question:
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        top_chunks = retrieve_relevant_chunks(question_embedding, chunk_embeddings, chunks, top_k=3, min_score=0.35)

        if not top_chunks:
            st.warning("🤖 Assistant: Sorry, I couldn't find that info. Want me to connect you with HR?")
        else:
            context = "\n".join(top_chunks)
            try:
                answer = generate_groq_answer(context, question)
                st.markdown("---")
                st.subheader("💬 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error from Groq API: {e}")
