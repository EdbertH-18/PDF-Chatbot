# 🧠 RAG Chatbot – HR Assistant Powered by PDF

A chatbot that serves as an HR assistant.  
It helps HR teams answer various questions from employees about company-related topics  
by extracting information from uploaded PDFs, using Retrieval-Augmented Generation (RAG).

The model reads your document, picks the most relevant part, and responds like a friendly admin on WhatsApp.  
No hallucination, no guessing — it only answers based on what’s actually written in the file.

---

## 📄 Recommended PDF Structure

To get the best results, use a clean, readable company document such as:

- Company Handbook  
- HR Policy Guidelines  
- Employee Onboarding PDF

Make sure the content inside the PDF is **text-based** (not image-scanned), and each section uses clear wording.  
Long, well-separated paragraphs work best for chunking and understanding.

---

## 🧠 Key Skills Showcased

- Retrieval-Augmented Generation (RAG)
- Semantic Search using SentenceTransformer
- PDF Parsing with PyMuPDF
- LLM Integration via Groq API (OpenAI-compatible)
- Streamlit for interactive app UI
