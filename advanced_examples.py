#!/usr/bin/env python3
"""
Advanced Milvus Examples
Demonstrates different search techniques and agent integration
"""

import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from datetime import datetime
import time


class AdvancedMilvusDB:
    """Advanced Milvus examples with multiple search techniques"""

    def __init__(self):
        self.host = "localhost"
        self.port = "19530"
        self.collection_name = "advanced_search"
        self.collection = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384

    def connect(self):
        """Connect to Milvus"""
        print(f"Connecting to Milvus at {self.host}:{self.port}...")
        connections.connect("default", host=self.host, port=self.port)
        print("✓ Connected!\n")

    def create_collection_with_metadata(self):
        """Create collection with metadata fields"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]

        schema = CollectionSchema(fields, description="Advanced search collection")
        self.collection = Collection(self.collection_name, schema)

        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print("✓ Collection with metadata created!\n")

    def load_sample_data_with_metadata(self):
        """Load sample data with categories and timestamps"""
        data = [
            ("Python is a versatile programming language for AI and web development", "programming", time.time() - 86400 * 1),
            ("Machine learning algorithms can predict future trends from data", "ai", time.time() - 86400 * 2),
            ("JavaScript frameworks like React make web development easier", "programming", time.time() - 86400 * 3),
            ("Deep neural networks power modern AI applications", "ai", time.time() - 86400 * 1),
            ("SpaceX is revolutionizing space travel with reusable rockets", "space", time.time() - 86400 * 5),
            ("The James Webb telescope reveals distant galaxies", "space", time.time() - 86400 * 2),
            ("Climate change affects global weather patterns", "climate", time.time() - 86400 * 1),
            ("Solar panels convert sunlight into clean energy", "climate", time.time() - 86400 * 4),
            ("Quantum computers use qubits for parallel processing", "technology", time.time() - 86400 * 3),
            ("Blockchain provides decentralized transaction ledgers", "technology", time.time() - 86400 * 6),
        ]

        texts = [item[0] for item in data]
        categories = [item[1] for item in data]
        timestamps = [int(item[2]) for item in data]
        embeddings = self.model.encode(texts).tolist()

        self.collection.insert([texts, categories, timestamps, embeddings])
        self.collection.load()
        print(f"✓ Loaded {len(texts)} documents with metadata\n")

    def basic_similarity_search(self, query, top_k=3):
        """Example 1: Basic similarity search"""
        print(f"{'='*80}")
        print(f"EXAMPLE 1: Basic Similarity Search")
        print(f"{'='*80}")
        print(f"Query: {query}\n")

        query_embedding = self.model.encode([query])

        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "category"]
        )

        print(f"Top {top_k} Results:")
        for i, hits in enumerate(results):
            for j, hit in enumerate(hits):
                similarity = 1 / (1 + hit.distance)
                print(f"\n{j+1}. Score: {similarity:.4f}")
                print(f"   Category: {hit.entity.get('category')}")
                print(f"   Text: {hit.entity.get('text')}")
        print()

    def filtered_search(self, query, category, top_k=3):
        """Example 2: Hybrid search with category filter"""
        print(f"{'='*80}")
        print(f"EXAMPLE 2: Filtered Search (Hybrid)")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Filter: category = '{category}'\n")

        query_embedding = self.model.encode([query])

        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            expr=f'category == "{category}"',  # Filter expression
            output_fields=["text", "category"]
        )

        print(f"Top {top_k} Results (filtered by category='{category}'):")
        for i, hits in enumerate(results):
            if len(hits) == 0:
                print("No results found for this category")
            for j, hit in enumerate(hits):
                similarity = 1 / (1 + hit.distance)
                print(f"\n{j+1}. Score: {similarity:.4f}")
                print(f"   Category: {hit.entity.get('category')}")
                print(f"   Text: {hit.entity.get('text')}")
        print()

    def multi_category_search(self, query, categories, top_k=3):
        """Example 3: Search across multiple categories"""
        print(f"{'='*80}")
        print(f"EXAMPLE 3: Multi-Category Search")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Categories: {categories}\n")

        query_embedding = self.model.encode([query])

        # Build filter for multiple categories
        category_filter = " || ".join([f'category == "{cat}"' for cat in categories])

        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            expr=category_filter,
            output_fields=["text", "category"]
        )

        print(f"Top {top_k} Results:")
        for i, hits in enumerate(results):
            for j, hit in enumerate(hits):
                similarity = 1 / (1 + hit.distance)
                print(f"\n{j+1}. Score: {similarity:.4f}")
                print(f"   Category: {hit.entity.get('category')}")
                print(f"   Text: {hit.entity.get('text')}")
        print()

    def batch_search(self, queries, top_k=2):
        """Example 4: Batch search multiple queries at once"""
        print(f"{'='*80}")
        print(f"EXAMPLE 4: Batch Search")
        print(f"{'='*80}")
        print(f"Queries: {queries}\n")

        query_embeddings = self.model.encode(queries)

        results = self.collection.search(
            data=query_embeddings.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "category"]
        )

        for query_idx, hits in enumerate(results):
            print(f"Query {query_idx+1}: '{queries[query_idx]}'")
            for j, hit in enumerate(hits):
                similarity = 1 / (1 + hit.distance)
                print(f"  {j+1}. Score: {similarity:.4f} | {hit.entity.get('text')[:60]}...")
            print()

    def query_by_id(self):
        """Example 5: Retrieve specific documents by ID"""
        print(f"{'='*80}")
        print(f"EXAMPLE 5: Query by ID")
        print(f"{'='*80}")

        # First, get some actual IDs from the collection
        all_docs = self.collection.query(
            expr="id >= 0",
            output_fields=["id"],
            limit=10
        )

        if len(all_docs) < 3:
            print("Not enough documents in collection\n")
            return

        # Get first 3 actual IDs
        actual_ids = [doc['id'] for doc in all_docs[:3]]
        print(f"Fetching documents with IDs: {actual_ids}\n")

        results = self.collection.query(
            expr=f"id in {actual_ids}",
            output_fields=["id", "text", "category"]
        )

        for doc in results:
            print(f"ID: {doc['id']}")
            print(f"Category: {doc['category']}")
            print(f"Text: {doc['text']}\n")

    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("✓ Disconnected from Milvus\n")


class RAGAgent:
    """Example 6: Simple RAG Agent"""

    def __init__(self, db):
        self.db = db

    def answer_question(self, question):
        """Retrieve context and format for LLM"""
        print(f"{'='*80}")
        print(f"EXAMPLE 6: RAG Agent - Question Answering")
        print(f"{'='*80}")
        print(f"Question: {question}\n")

        # Retrieve relevant context
        query_embedding = self.db.model.encode([question])
        results = self.db.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=3,
            output_fields=["text", "category"]
        )

        # Extract context
        context_docs = []
        print("Retrieved Context:")
        for i, hits in enumerate(results):
            for j, hit in enumerate(hits):
                text = hit.entity.get('text')
                context_docs.append(text)
                print(f"{j+1}. [{hit.entity.get('category')}] {text}")

        # Build prompt for LLM
        context = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(context_docs)])

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer: [This is where you would call your LLM - GPT-4, Claude, etc.]
"""

        print("\n" + "-" * 80)
        print("Prompt that would be sent to LLM:")
        print("-" * 80)
        print(prompt)
        print()


class SemanticCache:
    """Example 7: Semantic caching for LLM responses"""

    def __init__(self, db, similarity_threshold=0.85):
        self.db = db
        self.threshold = similarity_threshold
        self.cache_collection_name = "semantic_cache"
        self._setup_cache()

    def _setup_cache(self):
        """Setup cache collection"""
        if utility.has_collection(self.cache_collection_name):
            utility.drop_collection(self.cache_collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="response", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]

        schema = CollectionSchema(fields, description="Semantic cache")
        self.cache_collection = Collection(self.cache_collection_name, schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.cache_collection.create_index(field_name="embedding", index_params=index_params)
        self.cache_collection.load()

    def get(self, query):
        """Check cache for similar query"""
        query_embedding = self.db.model.encode([query])

        results = self.cache_collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=1,
            output_fields=["query", "response"]
        )

        for hits in results:
            if len(hits) > 0:
                hit = hits[0]
                similarity = 1 / (1 + hit.distance)

                if similarity >= self.threshold:
                    return {
                        'hit': True,
                        'similarity': similarity,
                        'cached_query': hit.entity.get('query'),
                        'response': hit.entity.get('response')
                    }

        return {'hit': False}

    def set(self, query, response):
        """Store query-response in cache"""
        embedding = self.db.model.encode([query])
        self.cache_collection.insert([[query], [response], embedding.tolist()])
        self.cache_collection.flush()  # Ensure data is persisted before search

    def demo(self):
        """Demonstrate semantic caching"""
        print(f"{'='*80}")
        print(f"EXAMPLE 7: Semantic Cache")
        print(f"{'='*80}\n")

        # Store some responses
        print("Caching responses...")
        self.set("What is Python?", "Python is a high-level programming language.")
        self.set("Tell me about machine learning", "Machine learning is a subset of AI.")
        print("✓ Cached 2 responses\n")

        # Test exact match
        print("Test 1: Exact query")
        query1 = "What is Python?"
        result1 = self.get(query1)
        print(f"Query: {query1}")
        print(f"Cache Hit: {result1['hit']}")
        if result1['hit']:
            print(f"Similarity: {result1['similarity']:.4f}")
            print(f"Response: {result1['response']}\n")

        # Test similar query (semantic match)
        print("Test 2: Similar query (semantic match)")
        query2 = "Can you explain Python to me?"
        result2 = self.get(query2)
        print(f"Query: {query2}")
        print(f"Cache Hit: {result2['hit']}")
        if result2['hit']:
            print(f"Similarity: {result2['similarity']:.4f}")
            print(f"Cached Query: {result2['cached_query']}")
            print(f"Response: {result2['response']}\n")

        # Test different query (cache miss)
        print("Test 3: Different query (cache miss)")
        query3 = "What about space exploration?"
        result3 = self.get(query3)
        print(f"Query: {query3}")
        print(f"Cache Hit: {result3['hit']}")
        if not result3['hit']:
            print("No similar query in cache - would call LLM\n")


def main():
    """Run all examples"""
    print("=" * 80)
    print("ADVANCED MILVUS EXAMPLES")
    print("=" * 80)
    print()

    # Setup
    db = AdvancedMilvusDB()
    db.connect()
    db.create_collection_with_metadata()
    db.load_sample_data_with_metadata()

    # Example 1: Basic similarity search
    db.basic_similarity_search("artificial intelligence and neural networks", top_k=3)

    # Example 2: Filtered search
    db.filtered_search("latest technology", category="programming", top_k=3)

    # Example 3: Multi-category search
    db.multi_category_search("scientific discoveries", categories=["space", "climate"], top_k=3)

    # Example 4: Batch search
    queries = ["programming languages", "space missions", "renewable energy"]
    db.batch_search(queries, top_k=2)

    # Example 5: Query by ID
    db.query_by_id()

    # Example 6: RAG Agent
    agent = RAGAgent(db)
    agent.answer_question("What programming languages are good for AI development?")

    # Example 7: Semantic Cache
    cache = SemanticCache(db)
    cache.demo()

    # Cleanup
    db.disconnect()

    print("=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
