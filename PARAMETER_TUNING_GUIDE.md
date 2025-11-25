# Vector Search Parameter Tuning Guide

## Table of Contents
1. [Search Parameters Deep Dive](#search-parameters-deep-dive)
2. [Distance Metrics Explained](#distance-metrics-explained)
3. [Index Parameters](#index-parameters)
4. [Industry Tools & Alternatives](#industry-tools--alternatives)
5. [Practical Tuning Recipes](#practical-tuning-recipes)

---

## Search Parameters Deep Dive

### 1. top_k (Number of Results)

**Definition**: The number of nearest neighbors to return from the vector search.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           top_k VISUALIZATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query: "How to reduce EC2 costs?"                                          │
│                                                                              │
│  Vector Space (2D simplified):                                               │
│                                                                              │
│         ○ "S3 pricing tiers"                                                │
│                                                                              │
│              ○ "Lambda cost optimization"                                   │
│                    ○ "EC2 rightsizing strategies"  ← top_k=1               │
│                  ● QUERY                                                    │
│                    ○ "Reduce compute costs"        ← top_k=2               │
│              ○ "EC2 instance savings"              ← top_k=3               │
│                                                                              │
│         ○ "Database backup strategies"                                      │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  top_k=1: Returns only "EC2 rightsizing strategies"                         │
│  top_k=3: Returns top 3 closest matches                                     │
│  top_k=5: Broader context, may include less relevant results                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**When to Use Different Values**:

| top_k | Use Case | Example |
|-------|----------|---------|
| 1-2 | Exact lookup, semantic cache | "What is AWS EC2?" |
| 3-5 | RAG retrieval (RECOMMENDED) | "How to optimize cloud costs?" |
| 5-10 | Broad research, multi-faceted questions | "Create a FinOps strategy" |
| 10-20 | Document clustering, comprehensive analysis | "Summarize all cost issues" |

**Code Example**:
```python
# Precise query - fewer results needed
results = collection.search(query_embedding, limit=2)  # top_k=2

# Complex query - need more context
results = collection.search(query_embedding, limit=7)  # top_k=7

# Dynamic top_k based on query complexity
def adaptive_top_k(query: str) -> int:
    """Adjust top_k based on query characteristics"""
    word_count = len(query.split())

    if word_count <= 5:
        return 2   # Simple, specific query
    elif word_count <= 12:
        return 5   # Moderate complexity
    else:
        return 8   # Complex, multi-part query
```

**Trade-offs**:
```
Higher top_k:
  ✅ More comprehensive context
  ✅ Less likely to miss relevant info
  ❌ More noise (irrelevant results)
  ❌ Higher token cost for LLM
  ❌ Slower processing

Lower top_k:
  ✅ Faster, cheaper
  ✅ More precise context
  ❌ May miss relevant information
  ❌ Fails on ambiguous queries
```

---

### 2. score_threshold (Similarity Cutoff)

**Definition**: Minimum similarity score required to include a result. Filters out weak matches.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCORE THRESHOLD VISUALIZATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Search Results (sorted by similarity):                                      │
│                                                                              │
│  Score │ Result                              │ Threshold Check              │
│  ──────┼─────────────────────────────────────┼────────────────────────────  │
│  0.92  │ "EC2 cost reduction strategies"     │ ✅ Pass (≥0.5)              │
│  0.85  │ "Rightsizing EC2 instances"         │ ✅ Pass (≥0.5)              │
│  0.71  │ "AWS compute pricing"               │ ✅ Pass (≥0.5)              │
│  0.48  │ "Cloud migration best practices"    │ ❌ Fail (<0.5)              │
│  0.35  │ "Kubernetes networking"             │ ❌ Fail (<0.5)              │
│  0.22  │ "Team standup meetings"             │ ❌ Fail (<0.5)              │
│        │                                     │                              │
│        │                    threshold = 0.5 ─┼─────────────────             │
│                                                                              │
│  With threshold=0.5: Returns 3 results                                       │
│  With threshold=0.3: Returns 5 results (more noise)                         │
│  With threshold=0.8: Returns 2 results (very strict)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Threshold Guidelines by Application**:

| Threshold | Application | Behavior |
|-----------|-------------|----------|
| 0.85-0.95 | Semantic caching | Near-exact match only |
| 0.70-0.85 | Duplicate detection | High confidence matches |
| 0.50-0.70 | RAG retrieval (default) | Balanced relevance |
| 0.30-0.50 | Exploratory search | Cast wide net |
| 0.10-0.30 | Clustering/analysis | Include weak associations |

**Code Example**:
```python
def search_with_threshold(
    collection,
    query_embedding,
    top_k: int = 10,
    score_threshold: float = 0.5
):
    """Search with dynamic threshold filtering"""

    # Get more results than needed, then filter
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k * 2,  # Fetch extra for filtering
        output_fields=["text", "metadata"]
    )

    filtered = []
    for hits in results:
        for hit in hits:
            # Convert L2 distance to similarity
            similarity = 1 / (1 + hit.distance)

            if similarity >= score_threshold:
                filtered.append({
                    "text": hit.entity.get("text"),
                    "score": similarity,
                    "metadata": hit.entity.get("metadata")
                })

    return filtered[:top_k]  # Return at most top_k


# Application-specific thresholds
THRESHOLDS = {
    "semantic_cache": 0.85,    # Only return if very similar
    "rag_retrieval": 0.50,     # Balanced
    "broad_search": 0.30,      # Include more context
    "strict_match": 0.75,      # High confidence only
}

# Usage
results = search_with_threshold(
    collection,
    query_embedding,
    score_threshold=THRESHOLDS["rag_retrieval"]
)
```

---

### 3. nprobe (Search Breadth)

**Definition**: Number of clusters to search during query time. Only applies to IVF-based indexes.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           nprobe EXPLAINED                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Index Creation (nlist=16 clusters):                                        │
│                                                                              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                   │
│  │ C1 │ │ C2 │ │ C3 │ │ C4 │ │ C5 │ │ C6 │ │ C7 │ │ C8 │                   │
│  │····│ │····│ │●●●●│ │····│ │····│ │····│ │····│ │····│                   │
│  │····│ │····│ │●●●●│ │····│ │····│ │····│ │····│ │····│                   │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘                   │
│                  ▲                                                           │
│  ┌────┐ ┌────┐ ┌┴───┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                   │
│  │ C9 │ │C10 │ │C11 │ │C12 │ │C13 │ │C14 │ │C15 │ │C16 │                   │
│  │····│ │····│ │●●●●│ │····│ │····│ │····│ │····│ │····│                   │
│  │····│ │····│ │●Q●●│ │····│ │····│ │····│ │····│ │····│                   │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘                   │
│                  ▲                                                           │
│                Query lands in cluster C11                                    │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  nprobe=1:  Search only C11          │ Fast, may miss neighbors in C3       │
│  nprobe=2:  Search C11 + C3          │ Better recall                        │
│  nprobe=4:  Search C11,C3,C10,C12    │ Good balance                         │
│  nprobe=16: Search ALL clusters      │ Same as brute force (slowest)        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**How nprobe Affects Performance**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    nprobe PERFORMANCE IMPACT                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dataset: 1M vectors, nlist=1024                                            │
│                                                                              │
│  nprobe │ Recall@10 │ Latency │ Vectors Scanned                            │
│  ───────┼───────────┼─────────┼─────────────────                            │
│    1    │   65%     │  0.5ms  │    ~1,000                                   │
│    4    │   82%     │  1.2ms  │    ~4,000                                   │
│    8    │   91%     │  2.1ms  │    ~8,000                                   │
│   16    │   96%     │  3.8ms  │   ~16,000                                   │
│   32    │   98%     │  7.2ms  │   ~32,000                                   │
│   64    │   99%     │ 14.1ms  │   ~64,000                                   │
│  1024   │  100%     │ 180ms   │ 1,000,000 (all)                             │
│                                                                              │
│  Recommended: nprobe = nlist/16 to nlist/8                                  │
│  For nlist=128: nprobe = 8 to 16                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example**:
```python
# Search parameters with nprobe tuning
def get_search_params(accuracy_level: str = "balanced"):
    """Get search params based on accuracy requirements"""

    configs = {
        "fast": {
            "metric_type": "L2",
            "params": {"nprobe": 4}    # ~80% recall, fastest
        },
        "balanced": {
            "metric_type": "L2",
            "params": {"nprobe": 16}   # ~95% recall, good speed
        },
        "accurate": {
            "metric_type": "L2",
            "params": {"nprobe": 64}   # ~99% recall, slower
        },
        "exact": {
            "metric_type": "L2",
            "params": {"nprobe": 1024}  # 100% recall, slowest
        }
    }

    return configs.get(accuracy_level, configs["balanced"])

# Usage
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=get_search_params("balanced"),
    limit=5
)
```

---

### 4. nlist (Index Clustering)

**Definition**: Number of cluster centroids created during index building. Determines how vectors are partitioned.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           nlist VISUALIZATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1M Vectors with different nlist values:                                    │
│                                                                              │
│  nlist=16 (Large clusters)         │  nlist=1024 (Small clusters)          │
│  ┌────────────────────────┐        │  ┌────────────────────────┐           │
│  │  ●●●●●    ●●●●●●●      │        │  │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │           │
│  │   ●●●●●   ●●●●●        │        │  │ │●││●││●││●││●││●││●│  │           │
│  │                        │        │  │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘  │           │
│  │  ●●●●●●   ●●●●●●●      │        │  │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐  │           │
│  │    ●●●●    ●●●●●       │        │  │ │●││●││●││●││●││●││●│  │           │
│  └────────────────────────┘        │  │ └─┘└─┘└─┘└─┘└─┘└─┘└─┘  │           │
│                                    │  └────────────────────────┘           │
│  ~62,500 vectors per cluster       │  ~1,000 vectors per cluster           │
│  Faster indexing                   │  Faster searching (less to scan)      │
│  Less memory                       │  More memory for centroids            │
│  Lower recall at low nprobe        │  Better recall at low nprobe          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**nlist Selection Rules**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         nlist SELECTION GUIDE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Rule of Thumb: nlist = 4 × √n  (where n = number of vectors)               │
│                                                                              │
│  Dataset Size │ Recommended nlist │ Vectors per Cluster                     │
│  ─────────────┼───────────────────┼─────────────────────                    │
│     10,000    │      128          │      ~78                                │
│     50,000    │      256          │     ~195                                │
│    100,000    │      512          │     ~195                                │
│    500,000    │     1024          │     ~488                                │
│  1,000,000    │     2048          │     ~488                                │
│  5,000,000    │     4096          │    ~1220                                │
│                                                                              │
│  Constraints:                                                                │
│  • Minimum: nlist ≥ 1                                                       │
│  • Maximum: nlist ≤ 65536 (Milvus limit)                                    │
│  • Should have: nlist < n/10 (at least 10 vectors per cluster)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Code Example**:
```python
import math

def calculate_optimal_nlist(num_vectors: int) -> int:
    """Calculate optimal nlist based on dataset size"""

    # 4√n rule
    optimal = int(4 * math.sqrt(num_vectors))

    # Apply constraints
    min_nlist = 16
    max_nlist = 65536

    # Ensure at least 10 vectors per cluster
    max_by_density = num_vectors // 10

    nlist = max(min_nlist, min(optimal, max_nlist, max_by_density))

    # Round to nearest power of 2 (optional, for memory efficiency)
    nlist = 2 ** round(math.log2(nlist))

    return nlist

# Examples
print(calculate_optimal_nlist(10_000))    # 128
print(calculate_optimal_nlist(100_000))   # 512
print(calculate_optimal_nlist(1_000_000)) # 4096
```

---

## Distance Metrics Explained

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DISTANCE METRICS COMPARISON                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Metric          │ Formula              │ Range        │ Similar When       │
│  ────────────────┼──────────────────────┼──────────────┼─────────────────── │
│  L2 (Euclidean)  │ √Σ(aᵢ - bᵢ)²        │ [0, ∞)       │ Distance → 0       │
│  IP (Inner Prod) │ Σ(aᵢ × bᵢ)          │ (-∞, ∞)      │ Value → Higher     │
│  Cosine          │ Σ(aᵢ×bᵢ)/(‖a‖×‖b‖)  │ [-1, 1]      │ Value → 1          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. L2 (Euclidean Distance)

**Definition**: Straight-line distance between two points in vector space.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        L2 EUCLIDEAN DISTANCE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Formula: d(a,b) = √[(a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²]                 │
│                                                                              │
│  Visual (2D example):                                                        │
│                                                                              │
│       ↑                                                                      │
│     5 │           ● B (4, 5)                                                │
│       │          /                                                           │
│     4 │         / d = √[(4-1)² + (5-2)²]                                   │
│       │        /    = √[9 + 9]                                              │
│     3 │       /     = √18 ≈ 4.24                                           │
│       │      /                                                               │
│     2 │     ● A (1, 2)                                                      │
│       │                                                                      │
│     1 │                                                                      │
│       └──────────────────────→                                              │
│         1   2   3   4   5                                                   │
│                                                                              │
│  Properties:                                                                 │
│  • Lower distance = More similar                                            │
│  • Zero distance = Identical vectors                                        │
│  • Sensitive to vector magnitude                                            │
│  • Most intuitive metric                                                    │
│                                                                              │
│  Best For:                                                                   │
│  ✅ Normalized embeddings (most common)                                     │
│  ✅ When magnitude matters equally to direction                             │
│  ✅ General-purpose similarity search                                       │
│                                                                              │
│  Code:                                                                       │
│  similarity = 1 / (1 + distance)  # Convert to 0-1 range                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Inner Product (IP)

**Definition**: Sum of element-wise products. Measures both direction similarity and magnitude.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INNER PRODUCT                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Formula: IP(a,b) = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ = Σ(aᵢ × bᵢ)              │
│                                                                              │
│  Visual (2D example):                                                        │
│                                                                              │
│       ↑                                                                      │
│       │     B (3, 4)                                                        │
│       │    ↗                                                                │
│       │   /                                                                 │
│       │  /   θ                                                              │
│       │ / ↗ A (4, 2)                                                       │
│       │/                                                                    │
│       └─────────────────→                                                   │
│                                                                              │
│  IP(A, B) = (4×3) + (2×4) = 12 + 8 = 20                                    │
│                                                                              │
│  For normalized vectors: IP = cos(θ)                                        │
│                                                                              │
│  Properties:                                                                 │
│  • Higher value = More similar (opposite of L2!)                            │
│  • Sensitive to magnitude (longer vectors → higher IP)                      │
│  • Equals cosine similarity when vectors are normalized                     │
│                                                                              │
│  Best For:                                                                   │
│  ✅ Maximum Inner Product Search (MIPS)                                     │
│  ✅ Recommendation systems                                                  │
│  ✅ When magnitude encodes importance                                       │
│  ✅ Pre-normalized embeddings (then IP = Cosine)                           │
│                                                                              │
│  Code:                                                                       │
│  # For normalized vectors                                                   │
│  similarity = (1 + ip_score) / 2  # Convert to 0-1 range                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Cosine Similarity

**Definition**: Measures the angle between two vectors, ignoring magnitude.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COSINE SIMILARITY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Formula: cos(θ) = (A · B) / (‖A‖ × ‖B‖)                                   │
│                  = Σ(aᵢ × bᵢ) / (√Σaᵢ² × √Σbᵢ²)                           │
│                                                                              │
│  Visual:                                                                     │
│                                                                              │
│       ↑                        ↑                        ↑                   │
│       │    B                   │  B                     │                   │
│       │   ↗                    │ ↗                      │ B                 │
│       │  /  θ small            │/  θ=90°                │↑                  │
│       │ ↗ A                    └──→ A                   └──→ A              │
│       └────→                                                                │
│                                                                              │
│    cos(θ) ≈ 0.95            cos(θ) = 0              cos(θ) = -1            │
│    Very similar             Unrelated               Opposite               │
│                                                                              │
│  Value Interpretation:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1.0      │  Identical direction                                    │   │
│  │  0.7-0.9  │  Similar (good match for RAG)                          │   │
│  │  0.3-0.7  │  Somewhat related                                       │   │
│  │  0.0      │  Perpendicular (unrelated)                             │   │
│  │  -1.0     │  Opposite direction                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Properties:                                                                 │
│  • Range: [-1, 1]                                                           │
│  • Magnitude-invariant (only measures direction)                            │
│  • Most popular for text embeddings                                         │
│                                                                              │
│  Best For:                                                                   │
│  ✅ Text similarity (different document lengths)                            │
│  ✅ When direction matters more than magnitude                              │
│  ✅ NLP embeddings (BERT, Sentence Transformers)                           │
│                                                                              │
│  Implementation Note:                                                        │
│  Milvus doesn't have native COSINE metric. Use IP with normalized vectors: │
│                                                                              │
│  # Normalize before inserting                                               │
│  normalized = embedding / np.linalg.norm(embedding)                         │
│  # Then use IP metric - equivalent to cosine similarity                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metric Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHICH METRIC SHOULD I USE?                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    Are your embeddings normalized?                           │
│                              │                                               │
│              ┌───────────────┴───────────────┐                              │
│              │                               │                               │
│             YES                              NO                              │
│              │                               │                               │
│              ▼                               ▼                               │
│    ┌─────────────────┐            ┌─────────────────┐                       │
│    │ Use L2 or IP    │            │ Does magnitude  │                       │
│    │ (equivalent for │            │ matter?         │                       │
│    │ normalized)     │            └────────┬────────┘                       │
│    └─────────────────┘                     │                                │
│                               ┌────────────┴────────────┐                   │
│                               │                         │                   │
│                              YES                        NO                  │
│                               │                         │                   │
│                               ▼                         ▼                   │
│                    ┌─────────────────┐      ┌─────────────────┐            │
│                    │ Use L2          │      │ Normalize first │            │
│                    │ (captures both  │      │ then use IP     │            │
│                    │ direction and   │      │ (= Cosine)      │            │
│                    │ magnitude)      │      │                 │            │
│                    └─────────────────┘      └─────────────────┘            │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Common Embedding Models and Recommended Metrics:                           │
│                                                                              │
│  Model                        │ Pre-normalized? │ Recommended Metric        │
│  ────────────────────────────┼─────────────────┼─────────────────────────  │
│  all-MiniLM-L6-v2            │ Yes             │ L2 or IP                  │
│  all-mpnet-base-v2           │ Yes             │ L2 or IP                  │
│  OpenAI text-embedding-3-*   │ Yes             │ Cosine (use IP)           │
│  Cohere embed-v3             │ Yes             │ Cosine (use IP)           │
│  BERT (raw)                  │ No              │ L2 after normalizing      │
│  Word2Vec                    │ No              │ L2 or Cosine              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Index Parameters

### Index Types Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INDEX TYPES COMPARISON                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Index      │ Build    │ Search  │ Memory  │ Recall │ Best For             │
│  ───────────┼──────────┼─────────┼─────────┼────────┼──────────────────────│
│  FLAT       │ O(1)     │ O(n)    │ Low     │ 100%   │ < 10K vectors        │
│  IVF_FLAT   │ O(n)     │ O(n/k)  │ Medium  │ 95%+   │ 10K - 1M vectors     │
│  IVF_SQ8    │ O(n)     │ O(n/k)  │ Low     │ 90%+   │ Memory constrained   │
│  IVF_PQ     │ O(n)     │ O(n/k)  │ V.Low   │ 85%+   │ > 1M, low memory     │
│  HNSW       │ O(nlogn) │ O(logn) │ High    │ 95%+   │ Low latency needs    │
│  SCANN      │ O(n)     │ O(n/k)  │ Medium  │ 95%+   │ Google-scale         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HNSW Parameters (Advanced)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HNSW INDEX PARAMETERS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HNSW = Hierarchical Navigable Small World                                  │
│                                                                              │
│  Structure (simplified):                                                     │
│                                                                              │
│  Layer 2:    ●───────────────────●                    (Few nodes, long jumps)
│              │                   │                                           │
│  Layer 1:    ●───────●───────────●───────●            (More nodes)          │
│              │       │           │       │                                   │
│  Layer 0:    ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●            (All nodes)           │
│                                                                              │
│  Parameters:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ M (edges per node):                                                    │ │
│  │ • Default: 16                                                          │ │
│  │ • Higher M = Better recall, More memory, Slower build                  │ │
│  │ • Typical: 8-64                                                        │ │
│  │                                                                        │ │
│  │ efConstruction (build-time search width):                              │ │
│  │ • Default: 256                                                         │ │
│  │ • Higher = Better graph quality, Slower build                          │ │
│  │ • Typical: 100-500                                                     │ │
│  │                                                                        │ │
│  │ ef (search-time width):                                                │ │
│  │ • Default: 64                                                          │ │
│  │ • Higher = Better recall, Slower search                                │ │
│  │ • Must be ≥ top_k                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Code Example:                                                               │
│  index_params = {                                                           │
│      "metric_type": "L2",                                                   │
│      "index_type": "HNSW",                                                  │
│      "params": {                                                            │
│          "M": 16,                # Connections per node                     │
│          "efConstruction": 256   # Build quality                            │
│      }                                                                      │
│  }                                                                          │
│                                                                              │
│  search_params = {                                                          │
│      "metric_type": "L2",                                                   │
│      "params": {"ef": 64}        # Search quality                           │
│  }                                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Industry Tools & Alternatives

### Vector Database Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE LANDSCAPE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Database     │ Type        │ Strengths              │ Best For             │
│  ─────────────┼─────────────┼────────────────────────┼──────────────────────│
│  Milvus       │ Purpose-    │ Scalable, GPU support, │ Large-scale prod     │
│               │ built       │ hybrid search          │ deployments          │
│               │             │                        │                      │
│  Pinecone     │ Managed     │ Zero ops, easy setup,  │ Quick start,         │
│               │ SaaS        │ good performance       │ managed preferred    │
│               │             │                        │                      │
│  Weaviate     │ Purpose-    │ GraphQL, modules,      │ Semantic search      │
│               │ built       │ built-in vectorizers   │ with schema          │
│               │             │                        │                      │
│  Qdrant       │ Purpose-    │ Rust performance,      │ High performance,    │
│               │ built       │ filtering, payloads    │ filtered search      │
│               │             │                        │                      │
│  ChromaDB     │ Embedded    │ Simple API, local,     │ Prototyping,         │
│               │             │ Python-native          │ small projects       │
│               │             │                        │                      │
│  pgvector     │ Extension   │ PostgreSQL native,     │ Existing Postgres    │
│               │             │ ACID, familiar SQL     │ infrastructure       │
│               │             │                        │                      │
│  Elasticsearch│ Search +    │ Full-text + vector,    │ Hybrid text +        │
│               │ Vector      │ existing ES expertise  │ semantic search      │
│               │             │                        │                      │
│  Redis Stack  │ Cache +     │ Ultra-low latency,     │ Real-time apps,      │
│               │ Vector      │ caching built-in       │ session similarity   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Embedding Model Options

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EMBEDDING MODEL COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OPEN SOURCE (Local)                                                        │
│  ───────────────────────────────────────────────────────────────────────── │
│  Model                    │ Dim   │ Speed  │ Quality │ Use Case            │
│  ─────────────────────────┼───────┼────────┼─────────┼─────────────────────│
│  all-MiniLM-L6-v2         │ 384   │ Fast   │ Good    │ General purpose     │
│  all-mpnet-base-v2        │ 768   │ Medium │ Better  │ Higher quality      │
│  e5-large-v2              │ 1024  │ Slow   │ Best    │ Benchmarks leader   │
│  bge-large-en-v1.5        │ 1024  │ Slow   │ Best    │ Multilingual        │
│  nomic-embed-text-v1.5    │ 768   │ Medium │ Great   │ Long context (8K)   │
│  gte-large                │ 1024  │ Slow   │ Best    │ Alibaba's model     │
│                                                                              │
│  COMMERCIAL APIs                                                             │
│  ───────────────────────────────────────────────────────────────────────── │
│  Provider       │ Model                  │ Dim   │ Price/1M tokens        │
│  ───────────────┼────────────────────────┼───────┼────────────────────────│
│  OpenAI         │ text-embedding-3-small │ 1536  │ $0.02                  │
│  OpenAI         │ text-embedding-3-large │ 3072  │ $0.13                  │
│  Cohere         │ embed-english-v3.0     │ 1024  │ $0.10                  │
│  Google         │ text-embedding-004     │ 768   │ $0.025                 │
│  Voyage AI      │ voyage-large-2         │ 1536  │ $0.12                  │
│  AWS Bedrock    │ Titan Embeddings       │ 1536  │ $0.10                  │
│                                                                              │
│  SPECIALIZED                                                                 │
│  ───────────────────────────────────────────────────────────────────────── │
│  • Code: CodeBERT, StarCoder embeddings                                     │
│  • Finance: FinBERT                                                         │
│  • Legal: Legal-BERT                                                        │
│  • Medical: BioBERT, PubMedBERT                                            │
│  • Multilingual: mBERT, LaBSE, SONAR                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Search Algorithms in Industry

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ANN ALGORITHMS COMPARISON                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Algorithm │ Used By          │ Pros                  │ Cons                │
│  ──────────┼──────────────────┼───────────────────────┼─────────────────────│
│  IVF       │ Faiss, Milvus    │ Simple, tunable       │ Recall vs speed     │
│            │                  │ balance               │ tradeoff            │
│            │                  │                       │                     │
│  HNSW      │ Qdrant, Weaviate │ Fast search, good     │ High memory,        │
│            │ Pinecone, Milvus │ recall                │ slow build          │
│            │                  │                       │                     │
│  ScaNN     │ Google           │ State-of-art for      │ Complex setup       │
│            │                  │ billion scale         │                     │
│            │                  │                       │                     │
│  DiskANN   │ Microsoft        │ SSD-optimized,        │ Needs fast SSD      │
│            │                  │ billion scale         │                     │
│            │                  │                       │                     │
│  Vamana    │ Research         │ Better than HNSW      │ Less mature         │
│            │                  │ in some cases         │                     │
│            │                  │                       │                     │
│  NSW       │ Legacy           │ Simple to implement   │ Superseded by HNSW  │
│                                                                              │
│  Industry Standard: HNSW (speed) or IVF_PQ (scale)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practical Tuning Recipes

### Recipe 1: FinOps RAG (Your Use Case)

```python
# finops_config.py

FINOPS_CONFIG = {
    # Embedding
    "model": "all-MiniLM-L6-v2",     # Fast, good for domain text
    "dimension": 384,

    # Index (for ~10K-100K FinOps documents)
    "index": {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 256}      # √100K × 4 ≈ 256
    },

    # Search
    "search": {
        "nprobe": 32,                 # nlist/8 for good recall
        "top_k": 5,                   # Enough context for RAG
        "score_threshold": 0.45       # Moderate threshold
    },

    # Multi-collection weights (optional)
    "collection_weights": {
        "finops_cost": 1.0,           # Primary domain
        "finops_budget": 0.9,
        "finops_utilization": 0.9,
        "finops_tools": 0.7           # Supplementary
    }
}
```

### Recipe 2: Semantic Cache (High Precision)

```python
CACHE_CONFIG = {
    "model": "all-mpnet-base-v2",     # Higher quality for matching
    "dimension": 768,

    "index": {
        "metric_type": "L2",
        "index_type": "HNSW",          # Fast exact lookup
        "params": {"M": 16, "efConstruction": 256}
    },

    "search": {
        "ef": 128,
        "top_k": 1,                    # Only need best match
        "score_threshold": 0.85        # High threshold for cache hit
    }
}
```

### Recipe 3: Large-Scale Production

```python
PRODUCTION_CONFIG = {
    "model": "text-embedding-3-small", # OpenAI for quality
    "dimension": 1536,

    "index": {
        "metric_type": "IP",            # For normalized OpenAI embeddings
        "index_type": "IVF_PQ",         # Memory efficient
        "params": {
            "nlist": 4096,              # Many clusters
            "m": 16,                    # PQ segments
            "nbits": 8                  # Bits per segment
        }
    },

    "search": {
        "nprobe": 128,                  # Higher for production quality
        "top_k": 10,
        "score_threshold": 0.5
    }
}
```

### Complete Tuning Code Example

```python
#!/usr/bin/env python3
"""
Parameter Tuning Helper for FinOps Agent
"""

from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class TuningConfig:
    """Centralized configuration for vector search tuning"""

    # Dataset info
    num_vectors: int
    avg_query_complexity: str  # "simple", "moderate", "complex"
    latency_requirement: str   # "realtime", "interactive", "batch"
    accuracy_requirement: str  # "approximate", "high", "exact"

    def get_index_params(self) -> Dict[str, Any]:
        """Calculate optimal index parameters"""

        # Choose index type based on scale
        if self.num_vectors < 10_000:
            return {
                "metric_type": "L2",
                "index_type": "FLAT",
                "params": {}
            }

        elif self.num_vectors < 1_000_000:
            nlist = int(4 * math.sqrt(self.num_vectors))
            nlist = min(max(nlist, 128), 4096)

            return {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": nlist}
            }

        else:
            # Large scale - use HNSW for speed or IVF_PQ for memory
            if self.latency_requirement == "realtime":
                return {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 256}
                }
            else:
                return {
                    "metric_type": "L2",
                    "index_type": "IVF_PQ",
                    "params": {"nlist": 4096, "m": 16, "nbits": 8}
                }

    def get_search_params(self) -> Dict[str, Any]:
        """Calculate optimal search parameters"""

        index_params = self.get_index_params()
        index_type = index_params["index_type"]

        if index_type == "FLAT":
            return {"metric_type": "L2", "params": {}}

        elif index_type in ["IVF_FLAT", "IVF_PQ"]:
            nlist = index_params["params"].get("nlist", 128)

            # nprobe based on accuracy requirement
            nprobe_ratios = {
                "approximate": 8,   # nlist/8
                "high": 4,          # nlist/4
                "exact": 1          # all clusters
            }
            ratio = nprobe_ratios.get(self.accuracy_requirement, 4)
            nprobe = max(nlist // ratio, 1)

            return {
                "metric_type": "L2",
                "params": {"nprobe": nprobe}
            }

        elif index_type == "HNSW":
            ef_values = {
                "approximate": 32,
                "high": 64,
                "exact": 256
            }
            return {
                "metric_type": "L2",
                "params": {"ef": ef_values.get(self.accuracy_requirement, 64)}
            }

        return {"metric_type": "L2", "params": {}}

    def get_top_k(self) -> int:
        """Calculate optimal top_k"""
        complexity_map = {
            "simple": 2,
            "moderate": 5,
            "complex": 8
        }
        return complexity_map.get(self.avg_query_complexity, 5)

    def get_score_threshold(self) -> float:
        """Calculate optimal score threshold"""
        accuracy_map = {
            "approximate": 0.3,
            "high": 0.5,
            "exact": 0.7
        }
        return accuracy_map.get(self.accuracy_requirement, 0.5)

    def summary(self) -> str:
        """Print configuration summary"""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    TUNING CONFIGURATION                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Dataset Size:     {self.num_vectors:,} vectors
║  Query Complexity: {self.avg_query_complexity}
║  Latency:          {self.latency_requirement}
║  Accuracy:         {self.accuracy_requirement}
╠══════════════════════════════════════════════════════════════════╣
║  INDEX PARAMS:     {self.get_index_params()}
║  SEARCH PARAMS:    {self.get_search_params()}
║  TOP_K:            {self.get_top_k()}
║  THRESHOLD:        {self.get_score_threshold()}
╚══════════════════════════════════════════════════════════════════╝
"""


# Example usage
if __name__ == "__main__":
    # FinOps Agent configuration
    config = TuningConfig(
        num_vectors=50_000,
        avg_query_complexity="moderate",
        latency_requirement="interactive",
        accuracy_requirement="high"
    )

    print(config.summary())
```

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PARAMETER QUICK REFERENCE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  top_k:      2-3 (precise) │ 5 (RAG default) │ 8-10 (broad)                │
│  threshold:  0.3 (loose)   │ 0.5 (balanced)  │ 0.7+ (strict)               │
│  nprobe:     nlist/16      │ nlist/8         │ nlist/4                     │
│  nlist:      4√n (rule of thumb)                                            │
│                                                                              │
│  Metric:     L2 (default)  │ IP (normalized) │ Cosine (text)               │
│  Index:      FLAT (<10K)   │ IVF (10K-1M)    │ HNSW (speed)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
