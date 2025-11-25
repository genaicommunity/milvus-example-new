# Milvus Vector Database - Complete Guide

A comprehensive example demonstrating Milvus vector database for document processing, semantic search, and AI agent integration.

## What's Included

This repository contains:
- **Basic Demo** - Simple vector search example
- **Advanced Examples** - 7 real-world techniques
- **Comprehensive Guide** - Deep dive into embeddings and search methods
- **Agent Integration** - RAG system and semantic caching examples

## Quick Start

### Prerequisites
- Docker Desktop (running)
- Python 3.8+
- pip

### Installation

1. **Start Milvus Database**
```bash
docker-compose up -d
```

2. **Install Dependencies**
```bash
pip install pymilvus sentence-transformers numpy
```

3. **Run Basic Demo**
```bash
python3 milvus_demo.py
```

4. **Run Advanced Examples**
```bash
python3 advanced_examples.py
```

## Project Files

```
milvus-example/
├── docker-compose.yml           # Milvus setup
├── milvus_demo.py              # Basic demo (start here)
├── advanced_examples.py        # 7 advanced techniques
├── sample_data.txt             # Sample documents
├── README.md                   # This file
└── EMBEDDINGS_GUIDE.md         # Comprehensive guide
```

## What You'll Learn

### From Basic Demo (milvus_demo.py)
- Connect to Milvus
- Create collections and indexes
- Store text as vectors
- Perform similarity search
- Retrieve relevant documents

**Output Example:**
```
Query: "Tell me about machine learning"
Result: Machine learning is a subset of AI... (Similarity: 0.6517)
```

### From Advanced Examples (advanced_examples.py)

**Example 1: Basic Similarity Search**
- Find semantically similar documents
- K-nearest neighbors search

**Example 2: Filtered Search (Hybrid)**
- Combine vector search + metadata filters
- Search within specific categories

**Example 3: Multi-Category Search**
- Search across multiple categories at once
- Complex boolean filters

**Example 4: Batch Search**
- Process multiple queries in parallel
- Efficient bulk operations

**Example 5: Query by ID**
- Fetch specific documents
- Direct retrieval without search

**Example 6: RAG Agent**
- Retrieval Augmented Generation
- Build context for LLM prompts
- Question-answering system

**Example 7: Semantic Cache**
- Cache LLM responses
- Match similar queries
- Save API costs

## Understanding the Components

### 1. How Embeddings Are Stored

```
Text → Embedding Model → Vector → Milvus
"Python is great"  →  [0.02, -0.14, 0.89, ...]  →  Database
```

**Storage Structure:**
```
Collection: document_search
├── id: 1, 2, 3, ...
├── text: "Python is...", "JavaScript is...", ...
├── embedding: [vector1], [vector2], ...
└── metadata: category, timestamp, etc.
```

### 2. Available Embedding Models

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| all-MiniLM-L6-v2 (used) | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Free |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Free |
| OpenAI text-3-small | 1536 | ⚡ | ⭐⭐⭐⭐⭐ | Paid |
| Cohere embed-v3 | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | Paid |

**Change Model Example:**
```python
# In milvus_demo.py or advanced_examples.py
model = SentenceTransformer('all-mpnet-base-v2')
dimension = 768  # Update schema dimension
```

### 3. Search Techniques

**Similarity Search (KNN)**
```python
results = collection.search(query_embedding, limit=5)
```
Use: Find most relevant documents

**Filtered Search (Hybrid)**
```python
results = collection.search(
    query_embedding,
    expr='category == "AI"',
    limit=5
)
```
Use: Search within constraints

**Range Search**
```python
results = collection.search(
    query_embedding,
    param={"radius": 1.0}
)
```
Use: Find all similar documents above threshold

### 4. Agent Integration

**RAG Pattern (Retrieval Augmented Generation)**
```python
# 1. Retrieve context from Milvus
context = milvus.search(user_question, top_k=5)

# 2. Build prompt
prompt = f"Context: {context}\n\nQuestion: {user_question}"

# 3. Call LLM
response = llm.generate(prompt)
```

**Benefits:**
- Reduces hallucinations
- Grounds responses in facts
- Updates without retraining

## Common Use Cases

### 1. Document Search
```python
# Store documents
db.process_file("documents.txt")

# Search
results = db.search("find information about X")
```

### 2. Question Answering
```python
# Retrieve relevant context
context = db.search(question, top_k=5)

# Pass to LLM
answer = llm.answer(question, context)
```

### 3. Semantic Recommendations
```python
# Find similar items
similar = db.search(current_item_text, top_k=10)
```

### 4. Chatbot Memory
```python
# Store conversations
db.insert(conversation_text)

# Retrieve relevant history
history = db.search(current_message, top_k=5)
```

### 5. Duplicate Detection
```python
# Check for similar content
matches = db.search(new_content, top_k=1)
if matches[0].similarity > 0.95:
    print("Duplicate found!")
```

## Advanced Topics (See EMBEDDINGS_GUIDE.md)

The comprehensive guide covers:
- Detailed storage architecture
- 15+ embedding models comparison
- Multiple retrieval techniques
- Agent integration patterns
- Reranking strategies
- Time-weighted search
- Diversity search (MMR)
- Production optimization

## API Reference

### MilvusVectorDB Class

```python
db = MilvusVectorDB()

# Connection
db.connect()
db.disconnect()

# Collection management
db.create_collection()
db.create_index()

# Data operations
db.process_file("file.txt")
db.search(query, top_k=5)
db.get_stats()
```

### AdvancedMilvusDB Class

```python
db = AdvancedMilvusDB()

# Enhanced search
db.basic_similarity_search(query, top_k=5)
db.filtered_search(query, category="AI")
db.multi_category_search(query, ["AI", "tech"])
db.batch_search([query1, query2, query3])
db.query_by_id([1, 2, 3])
```

### RAGAgent Class

```python
agent = RAGAgent(db)
response = agent.answer_question("What is Python?")
```

### SemanticCache Class

```python
cache = SemanticCache(db)

# Store
cache.set(query, response)

# Retrieve
cached = cache.get(similar_query)
if cached['hit']:
    return cached['response']
```

## Performance Tips

### 1. Choose Right Index
- **IVF_FLAT**: Balanced (used in demo)
- **HNSW**: Fastest for small datasets
- **IVF_SQ8**: Memory efficient
- **ANNOY**: Static datasets

### 2. Optimize Parameters
```python
# More clusters = faster but more memory
index_params = {"nlist": 128}  # Default
index_params = {"nlist": 256}  # Better for large datasets

# More probes = more accurate but slower
search_params = {"nprobe": 10}  # Default
search_params = {"nprobe": 20}  # More accurate
```

### 3. Batch Operations
```python
# Good: Batch insert
db.collection.insert([texts, embeddings])

# Avoid: Individual inserts
for text in texts:
    db.collection.insert([text, embedding])
```

### 4. Use Filters Early
```python
# Good: Filter during search
results = db.search(query, expr='category == "AI"')

# Avoid: Filter after search
results = db.search(query)
filtered = [r for r in results if r.category == "AI"]
```

## Troubleshooting

### Connection Failed
```
Error: failed to connect to localhost:19530
```
**Solution:** Ensure Docker containers are running
```bash
docker-compose ps  # Check status
docker-compose up -d  # Start if needed
```

### Out of Memory
```
Error: cannot allocate memory
```
**Solution:** Reduce batch size or use IVF_SQ8 index
```python
index_params = {"index_type": "IVF_SQ8"}  # Uses less memory
```

### Slow Search
```
Search taking too long
```
**Solution:** Increase nprobe or switch index
```python
# Faster but less accurate
search_params = {"nprobe": 5}

# Or use HNSW index for small datasets
index_params = {"index_type": "HNSW"}
```

### Low Quality Results
```
Results not relevant
```
**Solutions:**
1. Use better embedding model (all-mpnet-base-v2)
2. Increase top_k and rerank
3. Implement hybrid search with filters
4. Check data quality

## Docker Management

```bash
# Start Milvus
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs milvus

# Stop (keeps data)
docker-compose stop

# Stop and remove containers (keeps data)
docker-compose down

# Remove everything including data
docker-compose down -v
```

## Next Steps

1. **Try the basics**: Run `milvus_demo.py`
2. **Explore advanced**: Run `advanced_examples.py`
3. **Read the guide**: See `EMBEDDINGS_GUIDE.md`
4. **Customize**: Modify for your use case
5. **Scale up**: Deploy to production

## Production Checklist

- [ ] Choose appropriate embedding model
- [ ] Implement error handling and retries
- [ ] Add monitoring and logging
- [ ] Use connection pooling
- [ ] Implement authentication
- [ ] Set up backups
- [ ] Load test with production data
- [ ] Configure cluster mode for scale
- [ ] Implement rate limiting
- [ ] Add health checks

## Resources

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus GitHub](https://github.com/milvus-io/pymilvus)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Database Basics](https://milvus.io/docs/overview.md)

## Examples Output

When you run the demos, you'll see:

**Basic Demo:**
- Processes 15 documents
- Performs 3 similarity searches
- Shows relevance scores

**Advanced Examples:**
- 7 different search techniques
- RAG agent demonstration
- Semantic cache example
- Filtered and batch searches

## Contributing

Ideas for improvements:
- Add more embedding models
- Implement additional search techniques
- Create more agent patterns
- Add performance benchmarks
- Include production deployment guide

---

**Ready to get started?**
```bash
docker-compose up -d
python3 milvus_demo.py
```

Happy vector searching! 🚀
