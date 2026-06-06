---
title: "Retrieval Augmented Generation"
difficulty: intermediate
topic: llm
order: 9
estimatedTime: "45 minutes"
---

# Retrieval Augmented Generation (RAG)

## Overview

Retrieval Augmented Generation (RAG) addresses a fundamental limitation of LLMs: their knowledge is frozen at training time and stored imperfectly in model weights. RAG augments generation by retrieving relevant documents from an external knowledge base at inference time, grounding the model's output in verifiable sources. This reduces hallucinations, enables access to up-to-date or proprietary information, and makes the system's knowledge base auditable and updatable without retraining.

## The RAG Pipeline

A RAG system has three stages: **indexing**, **retrieval**, and **generation**.

**Indexing** (offline): Documents are chunked, embedded into vectors, and stored in a vector database. This is a one-time (or periodic) preprocessing step.

**Retrieval** (online): Given a user query, the system finds the most relevant chunks from the index.

**Generation** (online): The retrieved chunks are injected into the LLM's prompt as context, and the model generates an answer grounded in that context.

## Chunking Strategies

How you split documents matters significantly for retrieval quality:

- **Fixed-size chunking:** Split every N tokens with some overlap (e.g., 512 tokens with 50-token overlap). Simple but can break semantic units.
- **Recursive character splitting:** Split by paragraphs, then sentences, then characters -- preserving natural boundaries.
- **Semantic chunking:** Use embeddings to detect topic shifts and split at natural breakpoints.
- **Document-aware chunking:** Respect structure (headings, code blocks, tables) so chunks remain self-contained.

Typical chunk sizes range from 256 to 1024 tokens. Smaller chunks improve precision (less noise per chunk) but hurt recall (relevant information may be split across chunks).

## Embedding Models

Embedding models map text to dense vectors that capture semantic meaning. Popular choices include OpenAI's `text-embedding-3-small` (1536 dimensions), Cohere's `embed-v3`, and open-source models like `bge-large-en-v1.5` or `e5-large-v2`.

The similarity between a query vector $q$ and a document vector $d$ is typically measured by **cosine similarity**:

$$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \, \|d\|} = \frac{\sum_{i=1}^{n} q_i \, d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \cdot \sqrt{\sum_{i=1}^{n} d_i^2}}$$

This ranges from $-1$ (opposite) to $1$ (identical direction), with higher values indicating greater semantic similarity.

## Vector Databases

Vector databases are specialized for approximate nearest neighbor (ANN) search over high-dimensional embeddings:

- **Pinecone:** Managed service, serverless scaling, metadata filtering.
- **ChromaDB:** Open-source, lightweight, easy local development.
- **Weaviate:** Open-source, supports hybrid search natively.
- **Qdrant:** Open-source, Rust-based, high performance.
- **pgvector:** PostgreSQL extension for teams already using Postgres.

## Retrieval Methods

**Dense retrieval** uses embedding similarity (as above). It excels at semantic matching -- finding documents that mean the same thing even with different words.

**Sparse retrieval** uses traditional keyword matching. The classic BM25 scoring function ranks documents by term frequency:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

where $f(t, d)$ is the term frequency of term $t$ in document $d$, $|d|$ is the document length, $avgdl$ is the average document length, and $k_1$ (typically 1.2) and $b$ (typically 0.75) are tuning parameters. The IDF (inverse document frequency) is:

$$\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$

where $N$ is the total number of documents and $n(t)$ is the number containing term $t$.

**Hybrid retrieval** combines dense and sparse scores (e.g., with reciprocal rank fusion) to get the best of both worlds -- semantic understanding plus exact keyword matching.

## Re-Ranking

Initial retrieval is optimized for speed (searching millions of vectors). A **re-ranker** (e.g., Cohere Rerank, a cross-encoder model) then takes the top-K candidates and re-scores them with a more expensive but accurate model that jointly attends to the query and each document. This two-stage approach balances latency and accuracy.

## Code Example: Basic RAG Pipeline

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

client = OpenAI()

# Initialize ChromaDB with OpenAI embeddings
chroma_client = chromadb.Client()
ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)
collection = chroma_client.create_collection(
    name="knowledge_base",
    embedding_function=ef,
)

# Index documents (offline step)
documents = [
    "LoRA fine-tuning adds low-rank matrices to attention layers...",
    "Transformers use multi-head self-attention to process sequences...",
    "The Adam optimizer maintains per-parameter learning rates...",
]
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
)

# Retrieval + Generation (online step)
def rag_query(question: str, n_results: int = 3) -> str:
    # Retrieve relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    context_chunks = results["documents"][0]

    # Build prompt with retrieved context
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

    # Generate response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

answer = rag_query("How does LoRA reduce memory usage?")
print(answer)
```

## Evaluation

RAG systems require evaluation at both the retrieval and generation stages:

- **Retrieval metrics:** Recall@K (what fraction of relevant documents appear in the top K), Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG).
- **Generation metrics:**
  - **Faithfulness:** Does the answer only contain claims supported by the retrieved context? (Measures hallucination.)
  - **Relevance:** Does the answer address the user's question?
  - **Completeness:** Does the answer cover all relevant information from the retrieved documents?

Frameworks like RAGAS and TruLens automate these evaluations using LLM-as-judge approaches.

## Key Takeaways

1. RAG is often preferable to fine-tuning when you need factual accuracy over dynamic, updatable knowledge.
2. Chunking strategy has outsized impact on retrieval quality -- experiment with sizes and overlap.
3. Hybrid retrieval (dense + sparse) consistently outperforms either method alone.
4. Re-ranking is a cheap way to significantly boost precision in the final results.
5. Always evaluate faithfulness -- a RAG system that ignores its retrieved context is just a hallucinating LLM with extra steps.
