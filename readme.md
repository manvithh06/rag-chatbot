# Domain-Specific RAG Chatbot

A chatbot that answers questions about Machine Learning topics by 
retrieving relevant passages from a curated document corpus — 
not from the LLM's training data.

## Live Demo

🔗 [Try it here](https://rag-chatbot-376hj7kwh4wfrkajlvwg43.streamlit.app/)

## How It Works

```
User Question → Embed with all-MiniLM-L6-v2 
              → Search ChromaDB (cosine similarity)
              → Retrieve top-5 chunks 
              → Build contextualised prompt
              → Generate with Llama3-8b via Groq
              → Return answer with cited sources
```

## Corpus

15 Wikipedia articles covering ML fundamentals (~400+ chunks indexed):
Machine Learning, Deep Learning, Neural Networks, NLP, Transformers,
CNNs, RNNs, Gradient Descent, Overfitting, Cross-Validation,
Random Forest, SVMs, Reinforcement Learning, GANs, Transfer Learning.

**Total content:** ~200,000 characters, ~400 chunks at 500 chars each.

## 3 Demo Q&As

**Q1 (in-scope): "What is gradient descent?"**
> Gradient descent is an optimization algorithm used to minimize 
> a function by iteratively moving in the direction of steepest 
> descent as defined by the negative of the gradient. In machine 
> learning, it is used to update model parameters to minimize 
> the loss function. *(Source: Gradient descent)*

**Q2 (in-scope): "How do transformers work in NLP?"**
> Transformers use a self-attention mechanism that allows the model 
> to weigh the relevance of different words in a sequence when 
> encoding each word. Unlike RNNs, transformers process all tokens 
> in parallel, making them significantly faster to train on modern 
> hardware. *(Source: Transformer (machine learning model))*

**Q3 (out-of-scope): "What is the capital of France?"**
> I don't have information about that in my knowledge base. 
> My knowledge base covers Machine Learning topics only.




## Screenshots

[Add screenshots of your 3 demo Q&As here]

## Reflection

The hardest part was getting the grounding right — early versions would 
let the LLM fall back on its training knowledge when retrieved chunks 
were weak. The fix was a strict system prompt combined with low 
temperature (0.1), which forced the model to say "I don't know" 
rather than invent an answer.

Chunk size tuning was also non-obvious. At 200 chars, chunks were too 
small to contain complete thoughts. At 1000 chars, retrieval became 
imprecise. 500 chars with 50-char overlap hit the right balance for 
this corpus.

## Tech Stack
Python · Sentence-Transformers · ChromaDB · Groq (Llama3) · Streamlit
