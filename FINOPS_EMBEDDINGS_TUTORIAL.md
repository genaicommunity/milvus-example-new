# FinOps Agent: Embeddings & Vector Search Tutorial

## Table of Contents
1. [What Are Embeddings?](#what-are-embeddings)
2. [Architecture Overview](#architecture-overview)
3. [Key Parameters Explained](#key-parameters-explained)
4. [Fine-Tuning Guide](#fine-tuning-guide)
5. [Multi-Collection Strategy](#multi-collection-strategy)

---

## What Are Embeddings?

Embeddings are **numerical representations** of text that capture semantic meaning. Similar texts have similar embeddings (close in vector space).

### Simple Analogy

```
Text                          →  Embedding (simplified 3D)
─────────────────────────────────────────────────────────
"AWS cost analysis"           →  [0.8, 0.2, 0.1]
"Cloud spending report"       →  [0.75, 0.25, 0.15]  ← Similar!
"Kubernetes pod scheduling"   →  [0.1, 0.3, 0.9]    ← Different!
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PROCESS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   "What is my AWS spend?"                                        │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │   Tokenizer     │  Split into tokens                        │
│   │   ─────────     │  ["What", "is", "my", "AWS", "spend", "?"]│
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Transformer     │  Neural network processes tokens          │
│   │ Model           │  Captures context & relationships         │
│   │ (all-MiniLM)    │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Pooling Layer   │  Combine token embeddings                 │
│   │ (mean pooling)  │  into single vector                       │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   [0.023, -0.156, 0.089, ..., 0.234]   ← 384 dimensions         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### Complete FinOps Agent Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINOPS AGENT ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │   User Query     │
                         │ "How to reduce   │
                         │  EC2 costs?"     │
                         └────────┬─────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EMBEDDING LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SentenceTransformer('all-MiniLM-L6-v2')                            │    │
│  │  Input: "How to reduce EC2 costs?"                                   │    │
│  │  Output: [0.023, -0.156, 0.089, ..., 0.234]  (384 dimensions)       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Query Embedding
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MILVUS VECTOR DATABASE                                  │
│                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ finops_cost │ │finops_budget│ │finops_util  │ │finops_tools │           │
│  ├─────────────┤ ├─────────────┤ ├─────────────┤ ├─────────────┤           │
│  │ ● EC2 costs │ │ ● Budget    │ │ ● Under-    │ │ ● AWS Cost  │           │
│  │ ● RDS spend │ │   tracking  │ │   utilized  │ │   Explorer  │           │
│  │ ● Data      │ │ ● Alerts    │ │ ● Rightsize │ │ ● Kubecost  │           │
│  │   transfer  │ │ ● Forecast  │ │ ● RI cover  │ │ ● Infracost │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
│         │               │               │               │                   │
│         └───────────────┴───────────────┴───────────────┘                   │
│                              │                                               │
│                    Parallel Vector Search                                    │
│                    (ANN with IVF_FLAT index)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Search Results (Top-K from each collection)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT AGGREGATION                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ ContextBundle {                                                      │    │
│  │   cost_context: [                                                    │    │
│  │     {text: "Show EC2 costs by instance type", score: 0.89},         │    │
│  │     {text: "What are data transfer costs?", score: 0.72}            │    │
│  │   ],                                                                 │    │
│  │   budget_context: [...],                                             │    │
│  │   utilization_context: [                                             │    │
│  │     {text: "Which EC2 instances are underutilized?", score: 0.85}   │    │
│  │   ],                                                                 │    │
│  │   tools_context: [                                                   │    │
│  │     {text: "AWS Compute Optimizer: ML-powered...", score: 0.78}     │    │
│  │   ]                                                                  │    │
│  │ }                                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Rich Context
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROMPT BUILDER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ System: You are a FinOps expert...                                   │    │
│  │                                                                      │    │
│  │ ## Cost Information                                                  │    │
│  │ - Show EC2 costs by instance type (relevance: 0.89)                 │    │
│  │ - What are data transfer costs? (relevance: 0.72)                   │    │
│  │                                                                      │    │
│  │ ## Utilization Information                                           │    │
│  │ - Which EC2 instances are underutilized? (relevance: 0.85)          │    │
│  │                                                                      │    │
│  │ ## Recommended Tools                                                 │    │
│  │ - AWS Compute Optimizer: ML-powered rightsizing recommendations     │    │
│  │                                                                      │    │
│  │ User Question: How to reduce EC2 costs?                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    LLM Call      │
                         │  (Claude/GPT)    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Informed Answer  │
                         │ with specific    │
                         │ recommendations  │
                         └──────────────────┘
```

---

## Key Parameters Explained

### 1. Embedding Model Parameters

```python
# Model selection
model = SentenceTransformer('all-MiniLM-L6-v2')

# ┌────────────────────────────────────────────────────────────────┐
# │ MODEL COMPARISON                                                │
# ├────────────────────┬──────────┬─────────┬────────────────────── │
# │ Model              │ Dim      │ Speed   │ Quality              │
# ├────────────────────┼──────────┼─────────┼──────────────────────│
# │ all-MiniLM-L6-v2   │ 384      │ Fast    │ Good (recommended)   │
# │ all-mpnet-base-v2  │ 768      │ Medium  │ Better               │
# │ text-embedding-3-s │ 1536     │ API     │ Excellent            │
# │ text-embedding-3-l │ 3072     │ API     │ Best                 │
# └────────────────────┴──────────┴─────────┴──────────────────────┘
```

### 2. Index Parameters

```python
index_params = {
    "metric_type": "L2",        # Distance metric
    "index_type": "IVF_FLAT",   # Index algorithm
    "params": {"nlist": 128}    # Clustering parameter
}
```

#### Metric Types

```
┌─────────────────────────────────────────────────────────────────┐
│                      DISTANCE METRICS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  L2 (Euclidean)              │  IP (Inner Product)              │
│  ─────────────               │  ──────────────────              │
│  • Lower = more similar      │  • Higher = more similar         │
│  • Best for normalized       │  • Best for magnitude matters    │
│    embeddings                │                                  │
│  • Most common choice        │  • Use with cosine similarity    │
│                                                                  │
│  Formula:                    │  Formula:                        │
│  d = √Σ(a[i] - b[i])²       │  d = Σ(a[i] × b[i])              │
│                                                                  │
│       A ●                    │                                  │
│        \                     │       A ●───────→                │
│         \ d=0.5              │              θ                   │
│          \                   │       B ●───────→                │
│           ● B                │       IP = |A||B|cos(θ)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Index Types

```
┌─────────────────────────────────────────────────────────────────┐
│                       INDEX TYPES                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FLAT (Brute Force)                                             │
│  ├── Accuracy: 100% (exact)                                     │
│  ├── Speed: Slow for large datasets                             │
│  └── Use when: < 10,000 vectors                                 │
│                                                                  │
│  IVF_FLAT (Inverted File)                                       │
│  ├── Accuracy: ~95-99%                                          │
│  ├── Speed: Fast                                                │
│  ├── nlist: Number of clusters (√n to 4√n recommended)          │
│  └── Use when: 10K - 1M vectors                                 │
│                                                                  │
│  IVF_PQ (Product Quantization)                                  │
│  ├── Accuracy: ~90-95%                                          │
│  ├── Speed: Very fast, low memory                               │
│  └── Use when: > 1M vectors, memory constrained                 │
│                                                                  │
│  HNSW (Hierarchical Navigable Small World)                      │
│  ├── Accuracy: ~95-99%                                          │
│  ├── Speed: Very fast                                           │
│  ├── Memory: Higher                                             │
│  └── Use when: Need speed + accuracy                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Search Parameters

```python
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}  # Number of clusters to search
}

results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=top_k,              # Number of results
    expr='category == "cost"', # Optional filter
    output_fields=["text", "metadata"]
)
```

#### nprobe vs nlist

```
┌─────────────────────────────────────────────────────────────────┐
│                    nlist vs nprobe                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  nlist = 128 (clusters created during indexing)                 │
│                                                                  │
│     ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐           │
│     │ 1 │ │ 2 │ │ 3 │ │ 4 │ │...│ │126│ │127│ │128│           │
│     └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘           │
│       ▲     ▲     ▲     ▲                                       │
│       │     │     │     │                                       │
│       └─────┴─────┴─────┘                                       │
│              │                                                   │
│        nprobe = 4                                                │
│   (clusters searched at query time)                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Higher nprobe = Better accuracy, Slower search             │ │
│  │ Lower nprobe  = Faster search, May miss results            │ │
│  │                                                             │ │
│  │ Recommended: nprobe = nlist / 8 to nlist / 4               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fine-Tuning Guide

### 1. Score Threshold Tuning

```python
# Convert L2 distance to similarity score
similarity = 1 / (1 + distance)

# ┌─────────────────────────────────────────────────────────────────┐
# │ THRESHOLD GUIDELINES                                            │
# ├───────────────┬─────────────────────────────────────────────────┤
# │ Threshold     │ Use Case                                        │
# ├───────────────┼─────────────────────────────────────────────────┤
# │ 0.8 - 1.0     │ Near-exact match (semantic cache)               │
# │ 0.5 - 0.8     │ Strong relevance (RAG retrieval)                │
# │ 0.3 - 0.5     │ Moderate relevance (broad search)               │
# │ < 0.3         │ Weak match (might be noise)                     │
# └───────────────┴─────────────────────────────────────────────────┘
```

### 2. top_k Selection

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOP_K SELECTION                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  top_k = 1-3:  Precise answers, single topic                    │
│                "What is my EC2 cost?"                            │
│                                                                  │
│  top_k = 3-5:  Balanced context (RECOMMENDED for RAG)           │
│                "How can I optimize costs?"                       │
│                                                                  │
│  top_k = 5-10: Broad context, complex questions                 │
│                "Give me a complete FinOps strategy"              │
│                                                                  │
│  ⚠️  Too high top_k = noise + token cost                        │
│  ⚠️  Too low top_k = missing relevant context                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Index Parameter Tuning by Dataset Size

```python
def get_optimal_params(num_vectors: int) -> dict:
    """Get optimal index parameters based on dataset size"""

    if num_vectors < 10_000:
        # Small dataset - use exact search
        return {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {}
        }

    elif num_vectors < 1_000_000:
        # Medium dataset - IVF_FLAT
        nlist = int(4 * (num_vectors ** 0.5))  # 4√n rule
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": min(nlist, 4096)}
        }

    else:
        # Large dataset - IVF_PQ or HNSW
        return {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 256}
        }
```

---

## Multi-Collection Strategy

### When to Use Multiple Collections

```
┌─────────────────────────────────────────────────────────────────┐
│              COLLECTION STRATEGY DECISION TREE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Do you have different types of content?                         │
│           │                                                      │
│           ├── YES → Do they need different schemas?              │
│           │         │                                            │
│           │         ├── YES → MULTIPLE COLLECTIONS              │
│           │         │         (finops_cost, finops_tools, etc.) │
│           │         │                                            │
│           │         └── NO → Can you filter by field?            │
│           │                  │                                   │
│           │                  ├── YES → SINGLE COLLECTION         │
│           │                  │         + category field          │
│           │                  │                                   │
│           │                  └── NO → MULTIPLE COLLECTIONS       │
│           │                                                      │
│           └── NO → SINGLE COLLECTION                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### FinOps Multi-Collection Design

```
┌─────────────────────────────────────────────────────────────────┐
│                 FINOPS COLLECTION DESIGN                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    Schema:                                │
│  │  finops_cost     │    - id (INT64, PK)                       │
│  │  ──────────────  │    - text (VARCHAR)                       │
│  │  Cost queries    │    - metadata (JSON)                      │
│  │  & knowledge     │    - embedding (FLOAT_VECTOR[384])        │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌──────────────────┐    Same schema, different content:        │
│  │  finops_budget   │    Budget-specific queries                │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌──────────────────┐    Utilization & optimization             │
│  │  finops_util     │    queries                                │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌──────────────────┐    Different schema:                      │
│  │  finops_tools    │    - tool (VARCHAR)                       │
│  │  ──────────────  │    - capabilities (JSON array)           │
│  │  Tool metadata   │    - description embedding                │
│  └──────────────────┘                                           │
│                                                                  │
│  Why multiple collections?                                       │
│  1. Search specific domains independently                        │
│  2. Different relevance thresholds per domain                   │
│  3. Easier to update/rebuild individual domains                  │
│  4. Better organization and maintainability                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Parallel Search Pattern

```python
# Efficient: Search all collections in parallel
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_search(db, query, collections, top_k=3):
    """Search multiple collections concurrently"""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                db.search_collection,
                coll_name, query, top_k
            )
            for coll_name in collections
        ]
        results = await asyncio.gather(*tasks)

    return dict(zip(collections, results))
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EMBEDDING MODELS                                                │
│  ─────────────────                                               │
│  Fast:     all-MiniLM-L6-v2 (384d)                              │
│  Better:   all-mpnet-base-v2 (768d)                             │
│  Best:     text-embedding-3-large (3072d)                       │
│                                                                  │
│  INDEX TYPES                                                     │
│  ───────────                                                     │
│  < 10K vectors:    FLAT                                         │
│  10K - 1M vectors: IVF_FLAT (nlist=4√n)                         │
│  > 1M vectors:     HNSW or IVF_PQ                               │
│                                                                  │
│  SEARCH PARAMS                                                   │
│  ─────────────                                                   │
│  nprobe:    nlist/8 to nlist/4 (balance speed/accuracy)         │
│  top_k:     3-5 for RAG (balance context/noise)                 │
│  threshold: 0.5+ for strong relevance                           │
│                                                                  │
│  DISTANCE → SIMILARITY                                           │
│  ─────────────────────                                           │
│  L2:   similarity = 1 / (1 + distance)                          │
│  IP:   similarity = (1 + distance) / 2  [if normalized]         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Run the demo**: `python finops_agent.py`
2. **Add your data**: Expand `sample_queries/` with real FinOps knowledge
3. **Connect to LLM**: Replace the prompt builder with actual API calls
4. **Tune parameters**: Adjust thresholds based on your use case
5. **Scale up**: Use HNSW index for production workloads

---

*Tutorial created for FinOps Agent project*
