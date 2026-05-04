# Progress

A running log of what's done and what's next on **学点AI**. Update at the end of each work session.

## Status at a glance

| Track             | Lessons | Status                |
| ----------------- | ------: | --------------------- |
| AI Fundamentals   |      17 | All published         |
| Large Language Models |   9 | All published         |
| AI Agents (lessons)   |  15 | All published         |
| AI Agents (projects)  |   6 | All published         |
| AI for Biology        |       9 | All published         |
| AI for Physics        |      11 | All published         |
| **Total**             | **67** |                       |

All lessons currently render with KaTeX math and Mermaid diagrams.

## Lessons

### AI Fundamentals
- [x] 01 — What is Artificial Intelligence?
- [x] 02 — Machine Learning Basics
- [x] 03 — Deep Learning Foundations
- [x] 04 — CNNs for Image Processing
- [x] 05 — RNNs and LSTMs for Sequence Data
- [x] 06 — Transformers Architecture
- [x] 07 — AI Ethics and Bias
- [x] 08 — AI Safety and Alignment
- [x] 09 — Evaluating AI Systems
- [x] 10 — Future of AI
- [x] 11 — Real-World Applications of AI
- [x] 12 — AI Glossary and Key Concepts
- [x] 13 — Neural Networks: A Mathematical Perspective
- [x] 14 — Activation Functions Deep Dive
- [x] 15 — Regularization Techniques
- [x] 16 — Optimizers and Learning Rates
- [x] 17 — CNNs: Advanced Architectures

### Large Language Models
- [x] 01 — What are Large Language Models?
- [x] 02 — LLM Architecture Deep Dive
- [x] 03 — Training LLMs
- [x] 04 — Prompt Engineering Fundamentals
- [x] 05 — Tokenization and Embeddings
- [x] 06 — Context Windows and Attention
- [x] 07 — LLM Limitations and Hallucinations
- [x] 08 — Introduction to LLM Fine-Tuning
- [x] 09 — Retrieval Augmented Generation

### AI Agents
- [x] 01 — What are AI Agents?
- [x] 02 — Agent Architectures
- [x] 03 — Tool Use & Function Calling
- [x] 04 — Introduction to Agent Frameworks
- [x] 05 — Building Your First Simple Agent
- [x] 06 — Advanced Agent Patterns
- [x] 07 — Memory Systems
- [x] 08 — Tool Design & API Integration
- [x] 09 — Prompt Engineering for Agents
- [x] 10 — Error Handling & Robustness
- [x] 11 — Building Production Agents
- [x] 12 — Agent Evaluation & Testing
- [x] 13 — Security & Safety in AI Agents
- [x] 14 — Multi-Agent Collaboration
- [x] 15 — Agentic RAG Systems

### AI Agents — Projects
- [x] 01 — Q&A Agent with Tool Use
- [x] 02 — Multi-Agent Research System
- [x] 03 — RAG Agent
- [x] 04 — Production Customer Service Agent
- [x] 05 — Fine-Tune an LLM for Agent Tasks
- [x] 06 — Deploy an Agent to Production

### AI for Biology
- [x] 01 — Introduction to AI for Biology
- [x] 02 — Biological Data Representations
- [x] 03 — Key Datasets and Benchmarks in Computational Biology
- [x] 04 — Protein Structure Prediction
- [x] 05 — AlphaFold: Architecture and Impact
- [x] 06 — Genomics and Gene Expression with Deep Learning
- [x] 07 — Molecular Dynamics and Machine Learning Force Fields
- [x] 08 — Protein Design and Inverse Folding
- [x] 09 — Frontiers and Future Directions in AI for Biology

### AI for Physics
- [x] 01 — Introduction to AI for Physics
- [x] 02 — Physics-Informed Neural Networks
- [x] 03 — Neural Differential Equations
- [x] 04 — Classical vs Data-Driven Physics
- [x] 05 — Solving PDEs with Deep Learning
- [x] 06 — AI for Particle Physics
- [x] 07 — AI for Climate Modeling
- [x] 08 — Quantum Machine Learning
- [x] 09 — AI for Computational Chemistry
- [x] 10 — Neural Operators and Operator Learning
- [x] 11 — Frontiers in AI for Fundamental Physics

## Backlog / Ideas

Drop new lesson ideas here as they come up. Promote them into the lesson lists above when started.

- _(empty — add ideas)_

## Recent updates

Newest first. Use `YYYY-MM-DD` headers.

### 2026-05-04
- Added AI for Physics track: 11 lessons covering PINNs, neural differential equations, PDE solving with deep learning, particle physics AI, climate modeling, quantum ML, computational chemistry, neural operators, and frontiers in fundamental physics.

### 2026-05-03
- Added AI for Biology track: 9 lessons covering protein folding, AlphaFold, genomics, molecular dynamics, protein design, and future directions.

### 2026-05-02
- Added Mermaid diagram support and converted ASCII diagrams across all 33 lessons that had them.
- Added KaTeX math rendering (lazy-loaded).
- Switched markdown rendering from a homemade regex pass to `marked`.
- Fixed a few bugs: invisible cards on home page, lesson body not loading on hard refresh.

## Conventions for new lessons

1. **File**: drop the markdown into `public/content/<topic>/NN-slug.md` (zero-padded number).
2. **Register**: add an entry in `public/content/lessons.json` with `id`, `title`, `topic`, `topicPath`, `estimatedTime`, `difficulty`, `prerequisites`, `summary`, and `contentPath`.
3. **Frontmatter** in the markdown (currently stripped on render, but kept for source-of-truth): `---` block with `title`, `summary`, `difficulty`, `estimatedTime`.
4. **Diagrams**: use ` ```mermaid ` fenced blocks (flowchart / sequenceDiagram / stateDiagram-v2 / xychart-beta / quadrantChart). Add a bold `**Title**` line above each.
5. **Math**: use `$inline$` and `$$display$$`. KaTeX is loaded lazily on first lesson that contains math.
6. **Code samples**: tag the language (e.g. ` ```python `) so they're not mistaken for diagrams.
7. **Update this file**: tick the new lesson in the lesson lists and add a one-line note under "Recent updates".
