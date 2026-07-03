"""
ingest.py — Run this ONCE to build the vector store.
It fetches Wikipedia articles, chunks them, embeds them,
and saves everything to ChromaDB on disk.
"""

import os
import wikipedia
import chromadb
from chromadb.utils import embedding_functions

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuration ─────────────────────────────────────────
CHROMA_PATH = "chroma_db"
CORPUS_PATH = "corpus"
CHUNK_SIZE = 500       # tokens per chunk
CHUNK_OVERLAP = 50     # overlap between chunks (preserves context at boundaries)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, fast, high quality

# Topics to build our ML knowledge base from
# WHY these topics: they form a coherent domain — a reviewer can ask
# meaningful questions and see the bot answer vs refuse correctly
TOPICS = [
    "Machine learning",
    "Deep learning", 
    "Neural network (machine learning)",
    "Natural language processing",
    "Transformer (machine learning model)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Gradient descent",
    "Overfitting",
    "Cross-validation (statistics)",
    "Random forest",
    "Support vector machine",
    "Reinforcement learning",
    "Generative adversarial network",
    "Transfer learning",
]

def fetch_wikipedia_articles():
    """Download Wikipedia articles using the wikipedia package."""
    print("Fetching Wikipedia articles...")
    
    docs = []
    for topic in TOPICS:
        try:
            # Search and get the page
            page = wikipedia.page(topic, auto_suggest=False)
            
            # Save to corpus folder
            filename = topic.replace(' ', '_').replace('/', '_') + ".txt"
            filepath = os.path.join(CORPUS_PATH, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {page.title}\n\n{page.content}")
            
            docs.append({
                'text': page.content,
                'title': page.title,
                'url': page.url
            })
            print(f"  ✅ {page.title} ({len(page.content):,} chars)")
            
        except wikipedia.DisambiguationError as e:
            # If ambiguous, take first option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                docs.append({
                    'text': page.content,
                    'title': page.title,
                    'url': page.url
                })
                print(f"  ✅ {page.title} (via disambiguation)")
            except:
                print(f"  ❌ Skipped: {topic}")
                
        except Exception as e:
            print(f"  ❌ Failed: {topic} — {e}")
    
    print(f"\nFetched {len(docs)} articles")
    return docs

def chunk_documents(docs):
    """
    Split documents into chunks of ~500 tokens with 50-token overlap.
    
    WHY RecursiveCharacterTextSplitter:
    It tries to split on paragraphs first, then sentences, then words —
    preserving semantic units as much as possible rather than 
    blindly cutting at character 500.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,  # character-based approximation
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    metadatas = []
    ids = []
    
    for doc_idx, doc in enumerate(docs):
        doc_chunks = splitter.split_text(doc['text'])
        for chunk_idx, chunk in enumerate(doc_chunks):
            if len(chunk.strip()) < 50:  # skip tiny chunks
                continue
            chunks.append(chunk)
            metadatas.append({
                'title': doc['title'],
                'url': doc['url'],
                'chunk_index': chunk_idx,
                'doc_index': doc_idx
            })
            ids.append(f"doc{doc_idx}_chunk{chunk_idx}")
    
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"Average chunk size: {sum(len(c) for c in chunks)//len(chunks)} chars")
    return chunks, metadatas, ids

def build_vector_store(chunks, metadatas, ids):
    """
    Embed all chunks and store in ChromaDB.
    
    WHY all-MiniLM-L6-v2:
    - Completely free, runs on CPU
    - 384-dimensional embeddings — small enough to be fast
    - Trained on 1B+ sentence pairs — captures semantic meaning well
    - Industry standard for RAG prototypes
    """
    print(f"\nBuilding vector store with {EMBEDDING_MODEL}...")
    print("(This may take 2-5 minutes on first run — embeddings are computed locally)")
    
    # Sentence-transformer embedding function for ChromaDB
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Persistent ChromaDB client — saves to disk
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Delete collection if it exists (clean rebuild)
    try:
        client.delete_collection("ml_knowledge_base")
        print("Deleted existing collection for fresh rebuild")
    except:
        pass
    
    # Create collection
    collection = client.create_collection(
        name="ml_knowledge_base",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
    )
    
    # Add in batches (ChromaDB handles large inserts better in batches)
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            documents=chunks[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        print(f"  Embedded chunks {i} → {batch_end} / {len(chunks)}")
    
    print(f"\n✅ Vector store built successfully!")
    print(f"   Location: {CHROMA_PATH}/")
    print(f"   Total chunks indexed: {collection.count()}")
    return collection

if __name__ == "__main__":
    os.makedirs(CORPUS_PATH, exist_ok=True)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    docs = fetch_wikipedia_articles()
    chunks, metadatas, ids = chunk_documents(docs)
    collection = build_vector_store(chunks, metadatas, ids)
    
    print("\n🎉 Ingestion complete! Run: streamlit run app.py")