"""
rag_engine.py — The RAG pipeline.
Query → embed → retrieve → prompt → generate → return with sources.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ── Configuration ─────────────────────────────────────────
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_knowledge_base"
TOP_K = 5              # retrieve top 5 most relevant chunks
# NEW — currently active
LLM_MODEL = "llama-3.1-8b-instant"  # free on Groq
MAX_TOKENS = 1024

# ── System prompt — controls chatbot behaviour ─────────────
SYSTEM_PROMPT = """You are a knowledgeable assistant for a Machine Learning 
knowledge base. You answer questions using ONLY the context provided below.

Rules you must follow:
1. Answer ONLY from the provided context — do not use your own training knowledge
2. If the answer is not in the context, say exactly: 
   "I don't have information about that in my knowledge base."
3. Always cite which article your answer comes from
4. Be concise and precise — 2-4 sentences unless the question requires more
5. If the question is completely unrelated to ML/AI, politely decline

This strict grounding prevents hallucination and builds user trust."""

class RAGEngine:
    def __init__(self, api_key: str):
        """Initialise the RAG engine with ChromaDB + Groq."""
        
        # Load embedding function — same model used during ingestion
        # WHY same model: query and document embeddings must be in the same
        # vector space. Mixing models breaks semantic search.
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Connect to persisted ChromaDB
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef
        )
        
        # Groq client for LLM generation
        self.groq_client = Groq(api_key=api_key)
        
        print(f"RAG Engine ready — {self.collection.count()} chunks indexed")
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Embed the query and find the top-k most similar chunks.
        
        Returns dict with documents, metadatas, and distances.
        Distance < 0.3 = very relevant
        Distance > 0.7 = likely irrelevant
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        return results
    
    def build_prompt(self, query: str, retrieved_docs: dict) -> tuple:
        """
        Build the RAG prompt by inserting retrieved chunks as context.
        
        WHY this structure:
        - Numbered chunks help the model cite specific sources
        - Including article title helps with citation
        - Explicit instruction to use ONLY this context is critical
        """
        chunks = retrieved_docs['documents'][0]
        metas = retrieved_docs['metadatas'][0]
        distances = retrieved_docs['distances'][0]
        
        # Format context block
        context_parts = []
        sources = []
        
        for i, (chunk, meta, dist) in enumerate(zip(chunks, metas, distances)):
            context_parts.append(
                f"[Source {i+1}: {meta['title']}]\n{chunk}"
            )
            sources.append({
                'title': meta['title'],
                'url': meta['url'],
                'relevance': round(1 - dist, 3),  # convert distance to similarity
                'chunk': chunk[:150] + "..."
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        user_message = f"""Context from knowledge base:

{context}

---

Question: {query}

Answer based strictly on the context above. Cite your sources by article name."""
        
        return user_message, sources
    
    def generate(self, query: str) -> dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Build contextualised prompt
        3. Generate answer with LLM
        4. Return answer + sources
        """
        # Step 1: Retrieve
        retrieved = self.retrieve(query)
        
        # Step 2: Build prompt
        user_message, sources = self.build_prompt(query, retrieved)
        
        # Step 3: Generate
        response = self.groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1  # low temp = more factual, less creative
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'sources': sources,
            'query': query,
            'model': LLM_MODEL,
            'chunks_retrieved': len(sources)
        }