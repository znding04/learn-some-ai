---
title: "Agentic RAG Systems"
level: intermediate
topic: ai-agents
order: 15
estimatedTime: "45 minutes"
difficulty: advanced
prerequisites:
  - ai-agents-07
  - llm-05
summary: "Explore how Retrieval-Augmented Generation becomes agentic through query planning, iterative retrieval, self-evaluation, and multi-hop reasoning over documents."
---

# Agentic RAG Systems

## Overview

Standard RAG follows a simple pipeline: embed the query, retrieve top-k documents, stuff them into the context, and generate an answer. This works for straightforward factual questions but fails on complex queries requiring synthesis across multiple documents, disambiguation, or reasoning chains. Agentic RAG transforms retrieval into an iterative, self-directed process where an agent decides what to retrieve, evaluates whether retrieved information is sufficient, and adaptively refines its search strategy.

---

## From Passive to Agentic Retrieval

Traditional RAG is a single-shot pipeline:

1. User asks a question
2. System retrieves documents
3. LLM generates answer from context

Agentic RAG adds decision-making loops:

1. Agent **plans** what information it needs
2. Agent **formulates** retrieval queries (possibly multiple)
3. Agent **evaluates** retrieved documents for relevance and sufficiency
4. Agent **decides** whether to retrieve more, reformulate queries, or answer
5. Agent **synthesizes** across multiple retrieval rounds

---

## Relevance Scoring

The foundation of retrieval is measuring similarity between a query and documents. Cosine similarity between embedding vectors $\mathbf{q}$ (query) and $\mathbf{d}$ (document) is the standard metric:

$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \cdot \|\mathbf{d}\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \cdot \sqrt{\sum_{i=1}^{n} d_i^2}}$$

In agentic RAG, the agent also applies a learned relevance threshold $\tau$. A document is considered useful only if:

$$\text{sim}(\mathbf{q}, \mathbf{d}) > \tau$$

The agent can dynamically adjust $\tau$ based on the task -- lowering it when information is scarce, raising it when precision matters.

---

## Query Planning

Complex questions often require multiple sub-queries. The agent decomposes the original question into an information-gathering plan:

```python
from dataclasses import dataclass

@dataclass
class RetrievalPlan:
    original_query: str
    sub_queries: list[str]
    strategy: str  # "parallel", "sequential", "conditional"

def plan_queries(llm, question: str) -> RetrievalPlan:
    """Use the LLM to decompose a complex question into sub-queries."""
    prompt = f"""Given the question: "{question}"
    
    Decompose this into sub-queries needed to fully answer it.
    Return a JSON object with:
    - sub_queries: list of specific retrieval queries
    - strategy: "parallel" if independent, "sequential" if later 
      queries depend on earlier results
    """
    response = llm.generate(prompt)
    plan = parse_json(response)
    return RetrievalPlan(
        original_query=question,
        sub_queries=plan["sub_queries"],
        strategy=plan["strategy"]
    )
```

---

## The Agentic RAG Loop

The core of an agentic RAG system is an iterative retrieve-evaluate-decide loop:

```python
import numpy as np
from typing import Optional

class AgenticRAG:
    def __init__(self, llm, retriever, max_iterations: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.relevance_threshold = 0.75

    def answer(self, question: str) -> str:
        context = []
        plan = plan_queries(self.llm, question)
        queries_to_run = list(plan.sub_queries)

        for iteration in range(self.max_iterations):
            if not queries_to_run:
                break

            # Retrieve documents for current query
            current_query = queries_to_run.pop(0)
            docs = self.retriever.search(current_query, top_k=5)

            # Evaluate relevance of each document
            relevant_docs = self.evaluate_relevance(current_query, docs)
            context.extend(relevant_docs)

            # Self-evaluate: do we have enough information?
            assessment = self.assess_sufficiency(question, context)

            if assessment["sufficient"]:
                break
            elif assessment["needs_reformulation"]:
                new_query = assessment["reformulated_query"]
                queries_to_run.insert(0, new_query)
            # Otherwise continue with remaining planned queries

        return self.synthesize(question, context)

    def evaluate_relevance(self, query: str, docs: list[dict]) -> list[dict]:
        """Filter documents by relevance score and LLM judgment."""
        relevant = []
        for doc in docs:
            # Vector similarity check
            if doc["score"] < self.relevance_threshold:
                continue
            # LLM-based relevance verification
            judgment = self.llm.generate(
                f"Is this document relevant to '{query}'?\n"
                f"Document: {doc['text'][:500]}\n"
                f"Answer YES or NO with brief reason."
            )
            if "YES" in judgment.upper():
                relevant.append(doc)
        return relevant

    def assess_sufficiency(self, question: str, context: list[dict]) -> dict:
        """Ask the LLM whether gathered context is sufficient to answer."""
        context_text = "\n---\n".join(d["text"] for d in context)
        prompt = f"""Question: {question}

Retrieved context:
{context_text}

Can you fully answer the question with this context?
Respond with JSON:
- "sufficient": true/false
- "missing": what information is still needed (if any)
- "needs_reformulation": true/false
- "reformulated_query": better search query (if needed)
"""
        response = self.llm.generate(prompt)
        return parse_json(response)

    def synthesize(self, question: str, context: list[dict]) -> str:
        """Generate final answer from all gathered context."""
        context_text = "\n---\n".join(d["text"] for d in context)
        return self.llm.generate(
            f"Answer this question using the provided context.\n"
            f"Question: {question}\nContext:\n{context_text}"
        )
```

---

## Multi-Hop Reasoning

Some questions require chaining information across documents. For example: "What university did the CEO of the company that acquired Instagram attend?" requires:

1. Retrieve: who acquired Instagram (Facebook/Meta)
2. Retrieve: who is the CEO of Meta (Mark Zuckerberg)
3. Retrieve: where did Zuckerberg attend university (Harvard)

The agent's sequential strategy handles this naturally -- each retrieval result informs the next query. The information chain can be modeled as:

$$\text{answer} = f(d_1, d_2, \ldots, d_k)$$

where each document $d_{i+1}$ is retrieved using information extracted from $d_i$.

---

## Adaptive Retrieval Strategies

An agentic RAG system selects its retrieval approach based on question characteristics:

| Question Type | Strategy | Example |
|---------------|----------|---------|
| Factual lookup | Single retrieval | "When was Python created?" |
| Comparison | Parallel multi-query | "Compare React vs Vue performance" |
| Causal chain | Sequential retrieval | "Why did X lead to Y?" |
| Aggregation | Exhaustive retrieval | "List all products launched in 2024" |
| Ambiguous | Clarify then retrieve | "Tell me about Mercury" (planet? element?) |

---

## Self-Correction and Hallucination Detection

The agent can verify its own answer against retrieved evidence:

$$\text{grounding\_score} = \frac{|\text{claims supported by context}|}{|\text{total claims in answer}|}$$

If the grounding score falls below a threshold, the agent retrieves additional evidence or qualifies uncertain claims.

---

## Key Takeaways

- Agentic RAG replaces single-shot retrieval with iterative, self-directed search
- Query planning decomposes complex questions into focused sub-queries
- Self-evaluation prevents the system from answering with insufficient context
- Multi-hop reasoning chains information across sequential retrievals
- Adaptive strategies match retrieval approach to question complexity
- Grounding scores detect and prevent hallucination in final answers
