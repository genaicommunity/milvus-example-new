# Complete Guide: Embeddings, Storage, and Search Techniques

## Table of Contents
1. [How Embeddings Are Stored](#how-embeddings-are-stored)
2. [Available Embedding Models](#available-embedding-models)
3. [Fetching and Retrieving Embeddings](#fetching-and-retrieving-embeddings)
4. [Agent Integration and Context Passing](#agent-integration-and-context-passing)
5. [Different Search Techniques](#different-search-techniques)
6. [Advanced Usage Examples](#advanced-usage-examples)

---

## How Embeddings Are Stored

### Storage Architecture in Milvus

```python
# Collection Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

### Storage Process

1. **Text Input** → "Machine learning is powerful"
2. **Embedding Model** → Converts to 384-dimensional vector: `[0.023, -0.145, 0.891, ...]`
3. **Milvus Storage** → Stores vector + metadata in optimized format
4. **Indexing** → Creates searchable index structure (IVF_FLAT, HNSW, etc.)

### Physical Storage

```
Milvus Database
├── Collections (like SQL tables)
│   └── document_search
│       ├── Fields
│       │   ├── id: [1, 2, 3, ...]
│       │   ├── text: ["Python is...", "JavaScript is...", ...]
│       │   └── embedding: [[0.02, -0.14, ...], [0.15, 0.33, ...], ...]
│       └── Index (IVF_FLAT)
│           ├── Cluster 1: [vector_1, vector_5, vector_12]
│           ├── Cluster 2: [vector_2, vector_7, vector_9]
│           └── Cluster N: [...]
```

### Storage Formats

**In Memory:**
- Raw vectors: Float32 arrays
- Indexed: Quantized or compressed based on index type

**On Disk (MinIO/S3):**
- Segmented files for scalability
- Binary format for efficiency
- Automatic backup and replication

---

## Available Embedding Models

### 1. Sentence Transformers (Our Current Model)

#### all-MiniLM-L6-v2 (Used in Demo)
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```
- **Dimensions:** 384
- **Speed:** Very Fast
- **Use Case:** General purpose, semantic search
- **Quality:** Good balance of speed and accuracy
- **Best For:** Short texts, sentences, paragraphs

#### Other Popular Sentence Transformers

**all-mpnet-base-v2**
```python
model = SentenceTransformer('all-mpnet-base-v2')
dimension = 768  # Update in Milvus schema
```
- **Dimensions:** 768
- **Speed:** Medium
- **Quality:** Excellent
- **Best For:** High-quality semantic search

**paraphrase-MiniLM-L6-v2**
```python
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
dimension = 384
```
- **Dimensions:** 384
- **Speed:** Very Fast
- **Best For:** Paraphrase detection, question answering

**multi-qa-mpnet-base-dot-v1**
```python
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
dimension = 768
```
- **Dimensions:** 768
- **Best For:** Question-Answer pairs, FAQ systems

### 2. OpenAI Embeddings

```python
import openai

def get_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",  # or text-embedding-3-large
        input=text
    )
    return response['data'][0]['embedding']

# text-embedding-3-small: 1536 dimensions
# text-embedding-3-large: 3072 dimensions
# text-embedding-ada-002: 1536 dimensions (legacy)
```

**Pros:**
- State-of-the-art quality
- Consistent across use cases

**Cons:**
- Requires API calls (cost + latency)
- Not free

### 3. Cohere Embeddings

```python
import cohere

co = cohere.Client('your-api-key')

def get_cohere_embedding(text):
    response = co.embed(
        texts=[text],
        model='embed-english-v3.0'
    )
    return response.embeddings[0]

# embed-english-v3.0: 1024 dimensions
# embed-multilingual-v3.0: 1024 dimensions
```

### 4. HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch

# BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# bert-base-uncased: 768 dimensions
# roberta-base: 768 dimensions
# distilbert-base-uncased: 768 dimensions
```

### 5. Custom Domain-Specific Models

```python
# Medical domain
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Legal domain
model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')

# Code embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('microsoft/codebert-base')
```

### Comparison Table

| Model | Dimensions | Speed | Quality | Cost | Best Use Case |
|-------|-----------|-------|---------|------|---------------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Free | General purpose |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Free | High quality search |
| OpenAI text-3-small | 1536 | ⚡ | ⭐⭐⭐⭐⭐ | $$$ | Production apps |
| Cohere embed-v3 | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | $$$ | Multi-lingual |
| BERT-base | 768 | ⚡⚡ | ⭐⭐⭐ | Free | NLP tasks |

---

## Fetching and Retrieving Embeddings

### Method 1: Direct Query (What We Use)

```python
# Search with text - embedding generated on the fly
def search(self, query_text, top_k=3):
    # Generate embedding for query
    query_embedding = self.model.encode([query_text])

    # Search in Milvus
    results = self.collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    return results
```

### Method 2: Retrieve Specific Documents by ID

```python
def get_by_ids(self, ids):
    """Fetch specific documents by their IDs"""
    results = self.collection.query(
        expr=f"id in {ids}",
        output_fields=["id", "text", "embedding"]
    )
    return results

# Usage
docs = db.get_by_ids([1, 5, 10])
```

### Method 3: Retrieve with Filters

```python
# Add metadata field to schema
FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)

# Search with filter
def search_with_filter(self, query_text, category, top_k=3):
    query_embedding = self.model.encode([query_text])

    results = self.collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=f'category == "{category}"',  # Filter expression
        output_fields=["text", "category"]
    )
    return results

# Usage
results = db.search_with_filter("machine learning", category="AI")
```

### Method 4: Batch Retrieval

```python
def batch_search(self, queries, top_k=3):
    """Search multiple queries at once"""
    # Generate embeddings for all queries
    query_embeddings = self.model.encode(queries)

    # Batch search
    results = self.collection.search(
        data=query_embeddings.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    return results

# Usage
queries = ["AI trends", "space news", "programming tips"]
results = db.batch_search(queries)
```

### Method 5: Range Search

```python
def range_search(self, query_text, max_distance=1.0):
    """Find all documents within a distance threshold"""
    query_embedding = self.model.encode([query_text])

    results = self.collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={
            "metric_type": "L2",
            "params": {
                "nprobe": 10,
                "radius": max_distance  # Maximum distance
            }
        },
        limit=100,  # Max results
        output_fields=["text"]
    )
    return results
```

---

## Agent Integration and Context Passing

### Use Case 1: RAG (Retrieval Augmented Generation)

```python
def rag_agent_query(user_question):
    """
    Agent that retrieves context and passes to LLM
    """
    # 1. Retrieve relevant context from Milvus
    db = MilvusVectorDB()
    db.connect()

    # 2. Search for relevant documents
    results = db.search(user_question, top_k=5)

    # 3. Extract context
    context_docs = []
    for hits in results:
        for hit in hits:
            context_docs.append(hit.entity.get('text'))

    # 4. Build context string
    context = "\n\n".join([f"Document {i+1}: {doc}"
                          for i, doc in enumerate(context_docs)])

    # 5. Pass to LLM (e.g., OpenAI, Claude, etc.)
    prompt = f"""
    Context from knowledge base:
    {context}

    User Question: {user_question}

    Answer the question based on the context provided.
    """

    # 6. Call LLM (example with OpenAI)
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Usage
answer = rag_agent_query("What programming languages are good for AI?")
```

### Use Case 2: Multi-Agent System with Shared Memory

```python
class AgentMemory:
    """Shared memory for multiple agents using Milvus"""

    def __init__(self):
        self.db = MilvusVectorDB()
        self.db.connect()

    def store_observation(self, agent_name, observation):
        """Agent stores what it learned"""
        text = f"[{agent_name}] {observation}"
        embedding = self.db.model.encode([text])

        self.db.collection.insert([
            [text],
            embedding.tolist()
        ])

    def query_memory(self, agent_name, query, top_k=3):
        """Agent queries shared memory"""
        # Can filter by agent or search all
        results = self.db.search(query, top_k=top_k)
        return results

# Usage
memory = AgentMemory()

# Agent 1 stores info
memory.store_observation("ResearchAgent", "Python is widely used in AI research")

# Agent 2 queries
results = memory.query_memory("CodingAgent", "What language for AI?")
```

### Use Case 3: Conversational Agent with Context

```python
class ConversationalAgent:
    """Agent that maintains conversation context"""

    def __init__(self):
        self.db = MilvusVectorDB()
        self.db.connect()
        self.conversation_history = []

    def add_to_context(self, role, message):
        """Store conversation turn"""
        text = f"{role}: {message}"
        self.conversation_history.append(text)

        # Store in Milvus for long-term memory
        embedding = self.db.model.encode([text])
        self.db.collection.insert([[text], embedding.tolist()])

    def get_relevant_context(self, current_query, top_k=5):
        """Retrieve relevant past conversations"""
        results = self.db.search(current_query, top_k=top_k)

        context = []
        for hits in results:
            for hit in hits:
                context.append(hit.entity.get('text'))

        return context

    def respond(self, user_message):
        """Generate response with context"""
        # Add current message
        self.add_to_context("User", user_message)

        # Get relevant context
        relevant_context = self.get_relevant_context(user_message)

        # Build prompt with context
        prompt = f"""
        Relevant conversation history:
        {chr(10).join(relevant_context)}

        Current message: {user_message}

        Respond appropriately.
        """

        # Call LLM here...
        response = "..." # Your LLM call

        self.add_to_context("Assistant", response)
        return response
```

### Use Case 4: Tool-Using Agent with Knowledge Base

```python
class ToolAgent:
    """Agent that uses Milvus as a tool for information retrieval"""

    def __init__(self):
        self.db = MilvusVectorDB()
        self.db.connect()
        self.tools = {
            "search_knowledge": self.search_knowledge,
            "store_fact": self.store_fact
        }

    def search_knowledge(self, query):
        """Tool: Search knowledge base"""
        results = self.db.search(query, top_k=3)
        facts = []
        for hits in results:
            for hit in hits:
                facts.append(hit.entity.get('text'))
        return facts

    def store_fact(self, fact):
        """Tool: Store new fact"""
        embedding = self.db.model.encode([fact])
        self.db.collection.insert([[fact], embedding.tolist()])
        return "Fact stored successfully"

    def execute(self, task):
        """Main agent execution loop"""
        # Agent decides which tool to use based on task
        if "search" in task.lower() or "find" in task.lower():
            return self.search_knowledge(task)
        elif "remember" in task.lower() or "store" in task.lower():
            return self.store_fact(task)
        else:
            return "Task not understood"

# Usage
agent = ToolAgent()
result = agent.execute("search for information about machine learning")
```

---

## Different Search Techniques

### 1. Similarity Search (K-Nearest Neighbors)

**What we use in the demo**

```python
def knn_search(query_text, k=5):
    """Find K most similar documents"""
    query_embedding = model.encode([query_text])

    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["text"]
    )
    return results
```

**Best For:** Finding most relevant documents

### 2. Range Search

```python
def range_search(query_text, max_distance=1.0):
    """Find all documents within distance threshold"""
    query_embedding = model.encode([query_text])

    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={
            "metric_type": "L2",
            "params": {
                "radius": max_distance,
                "range_filter": 0.0  # Minimum distance
            }
        },
        limit=100,
        output_fields=["text"]
    )
    return results
```

**Best For:** Finding all similar documents above a quality threshold

### 3. Hybrid Search (Vector + Scalar Filtering)

```python
def hybrid_search(query_text, category, date_after, k=5):
    """Combine vector search with metadata filters"""
    query_embedding = model.encode([query_text])

    # Build filter expression
    filter_expr = f'category == "{category}" && date > {date_after}'

    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,
        expr=filter_expr,  # Scalar filter
        output_fields=["text", "category", "date"]
    )
    return results
```

**Best For:** Combining semantic similarity with business logic

### 4. Multi-Vector Search

```python
def multi_vector_search(title, content, k=5):
    """Search using multiple vectors (e.g., title + content)"""
    # Generate separate embeddings
    title_embedding = model.encode([title])
    content_embedding = model.encode([content])

    # Weighted combination
    combined = 0.3 * title_embedding + 0.7 * content_embedding

    results = collection.search(
        data=combined.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["text"]
    )
    return results
```

**Best For:** Documents with multiple semantic components

### 5. Reranking Search

```python
def reranking_search(query_text, k=20, final_k=5):
    """Two-stage search with reranking"""
    # Stage 1: Fast approximate search
    query_embedding = model.encode([query_text])

    initial_results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,  # Get more candidates
        output_fields=["text"]
    )

    # Stage 2: Rerank with cross-encoder or better model
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    candidates = []
    for hits in initial_results:
        for hit in hits:
            candidates.append(hit.entity.get('text'))

    # Score all candidates
    pairs = [[query_text, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    # Sort and return top final_k
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:final_k]
```

**Best For:** Maximum accuracy, production systems

### 6. Diversity Search (MMR - Maximal Marginal Relevance)

```python
def diversity_search(query_text, k=10, lambda_param=0.5):
    """Return diverse results, not just similar ones"""
    query_embedding = model.encode([query_text])

    # Get candidates
    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k * 2,  # Get more candidates
        output_fields=["text", "embedding"]
    )

    # MMR algorithm
    selected = []
    candidates = []

    for hits in results:
        for hit in hits:
            candidates.append({
                'text': hit.entity.get('text'),
                'embedding': hit.entity.get('embedding')
            })

    # Select first (most similar)
    selected.append(candidates[0])
    candidates.pop(0)

    # Iteratively select diverse documents
    while len(selected) < k and candidates:
        best_score = -float('inf')
        best_idx = 0

        for idx, candidate in enumerate(candidates):
            # Similarity to query
            query_sim = cosine_similarity(
                query_embedding,
                candidate['embedding']
            )

            # Max similarity to already selected
            max_sim_to_selected = max([
                cosine_similarity(candidate['embedding'], s['embedding'])
                for s in selected
            ])

            # MMR score
            score = lambda_param * query_sim - (1 - lambda_param) * max_sim_to_selected

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(candidates[best_idx])
        candidates.pop(best_idx)

    return selected
```

**Best For:** Avoiding redundant results

### 7. Time-Weighted Search

```python
def time_weighted_search(query_text, k=5, recency_weight=0.3):
    """Boost recent documents"""
    from datetime import datetime
    import time

    query_embedding = model.encode([query_text])

    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k * 2,
        output_fields=["text", "timestamp"]
    )

    # Rerank with time decay
    current_time = time.time()
    scored_results = []

    for hits in results:
        for hit in hits:
            similarity = 1 / (1 + hit.distance)

            # Time decay (exponential)
            age_seconds = current_time - hit.entity.get('timestamp')
            age_days = age_seconds / 86400
            time_score = np.exp(-recency_weight * age_days)

            # Combined score
            final_score = 0.7 * similarity + 0.3 * time_score

            scored_results.append({
                'text': hit.entity.get('text'),
                'score': final_score
            })

    # Sort and return top k
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    return scored_results[:k]
```

**Best For:** News, social media, time-sensitive content

---

## Advanced Usage Examples

### Complete RAG System

```python
class RAGSystem:
    """Production-ready RAG system"""

    def __init__(self, collection_name="rag_knowledge"):
        self.db = MilvusVectorDB()
        self.db.collection_name = collection_name
        self.db.connect()
        self.db.create_collection()
        self.db.create_index()

    def ingest_documents(self, documents):
        """Bulk ingest documents"""
        texts = []
        embeddings = []

        for doc in documents:
            texts.append(doc)
            embedding = self.db.model.encode([doc])[0]
            embeddings.append(embedding.tolist())

        self.db.collection.insert([texts, embeddings])
        self.db.collection.load()
        print(f"Ingested {len(documents)} documents")

    def query(self, question, top_k=5):
        """Query with context retrieval"""
        results = self.db.search(question, top_k=top_k)

        context_docs = []
        for hits in results:
            for hit in hits:
                context_docs.append(hit.entity.get('text'))

        return {
            'question': question,
            'context': context_docs,
            'prompt': self._build_prompt(question, context_docs)
        }

    def _build_prompt(self, question, context):
        """Build LLM prompt"""
        context_str = "\n\n".join([
            f"[{i+1}] {doc}" for i, doc in enumerate(context)
        ])

        return f"""
        Use the following context to answer the question.

        Context:
        {context_str}

        Question: {question}

        Answer:
        """

# Usage
rag = RAGSystem()
rag.ingest_documents([
    "Python is great for AI development",
    "Machine learning requires large datasets",
    # ... more documents
])

result = rag.query("What language is good for AI?")
print(result['prompt'])
```

### Semantic Cache

```python
class SemanticCache:
    """Cache LLM responses using semantic similarity"""

    def __init__(self, similarity_threshold=0.9):
        self.db = MilvusVectorDB()
        self.db.connect()
        self.threshold = similarity_threshold

    def get(self, query):
        """Check if similar query exists in cache"""
        results = self.db.search(query, top_k=1)

        for hits in results:
            for hit in hits:
                similarity = 1 / (1 + hit.distance)

                if similarity >= self.threshold:
                    # Cache hit
                    return hit.entity.get('response')

        # Cache miss
        return None

    def set(self, query, response):
        """Store query-response pair"""
        embedding = self.db.model.encode([query])

        # Store query and response
        text = f"Q: {query}\nA: {response}"
        self.db.collection.insert([[text], embedding.tolist()])

# Usage
cache = SemanticCache()

# First query (cache miss, call LLM)
response = cache.get("What is Python?")
if response is None:
    response = call_expensive_llm("What is Python?")
    cache.set("What is Python?", response)

# Similar query (cache hit, no LLM call needed)
response = cache.get("Tell me about Python")  # Returns cached response!
```

---

## Summary

### Quick Reference

**Store Embeddings:**
```python
embedding = model.encode([text])
collection.insert([[text], embedding.tolist()])
```

**Fetch Embeddings:**
```python
results = collection.search(query_embedding, limit=k)
```

**Agent Context:**
```python
context = retrieve_from_milvus(user_query)
llm_response = llm.generate(context + user_query)
```

**Best Practices:**
1. Choose embedding model based on your domain
2. Use hybrid search for production systems
3. Implement reranking for best accuracy
4. Cache embeddings to save compute
5. Monitor and optimize index parameters
6. Use metadata filtering for complex queries

---

**Next Steps:**
- Experiment with different embedding models
- Implement hybrid search with filters
- Build a RAG system for your use case
- Integrate with LLM agents
- Optimize for production scale
