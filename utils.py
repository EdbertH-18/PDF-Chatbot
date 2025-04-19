import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def load_pdf(file) -> list[str]:
    """Extract and split PDF text into paragraphs"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    chunks = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    return chunks

def embed_chunks(chunks: list[str], embedder: SentenceTransformer) -> torch.Tensor:
    """Convert text chunks into embedding vectors"""
    return embedder.encode(chunks, convert_to_tensor=True)

def retrieve_relevant_chunks(
    question_embedding: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    chunks: list[str],
    top_k: int = 3,
    min_score: float = 0.35
) -> list[str]:
    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    sorted_indices = torch.argsort(similarities, descending=True)

    filtered_chunks = []
    for idx in sorted_indices:
        score = similarities[idx].item()
        if score < min_score:
            break
        filtered_chunks.append(chunks[idx])
        if len(filtered_chunks) == top_k:
            break

    return filtered_chunks

def generate_groq_answer(context: str, question: str) -> str:
    prompt = (
        "You are a friendly HR assistant chatting on WhatsApp.\n"
        "Your job is to help employees understand company rules.\n"
        "Only answer questions if the information is found in the provided context below.\n"
        "Do NOT guess or make up anything — just use the context.\n"
        "If you don't find the answer, say: 'Sorry, I couldn't find that info in the handbook. Want me to connect you with HR?'\n"
        "Treat abbreviations like:\n"
        "- WFH = Work From Home\n"
        "- PTO = Paid Time Off\n"
        "- HR = Human Resources\n\n"
        f"Context:\n{context}\n\n"
        f"User: {question}\n"
        "You:"
    )

    response = client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()
