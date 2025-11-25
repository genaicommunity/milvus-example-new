#!/usr/bin/env python3
"""
Simple Milvus Vector Database Example
Demonstrates file processing and similarity search
"""

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer


class MilvusVectorDB:
    def __init__(self, host="localhost", port="19530"):
        """Initialize connection to Milvus"""
        self.host = host
        self.port = port
        self.collection_name = "document_search"
        self.collection = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2 model

    def connect(self):
        """Connect to Milvus server"""
        print(f"Connecting to Milvus at {self.host}:{self.port}...")
        connections.connect("default", host=self.host, port=self.port)
        print("✓ Connected successfully!")

    def create_collection(self):
        """Create a collection (table) in Milvus"""
        # Drop collection if it exists
        if utility.has_collection(self.collection_name):
            print(f"Dropping existing collection '{self.collection_name}'...")
            utility.drop_collection(self.collection_name)

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]

        schema = CollectionSchema(fields, description="Document search collection")

        print(f"Creating collection '{self.collection_name}'...")
        self.collection = Collection(self.collection_name, schema)
        print("✓ Collection created!")

    def create_index(self):
        """Create an index for efficient similarity search"""
        print("Creating index...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        print("✓ Index created!")

    def process_file(self, file_path):
        """Process a text file and store in Milvus"""
        print(f"\nProcessing file: {file_path}")

        # Read file
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Found {len(lines)} documents")

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(lines)

        # Prepare data
        data = [
            lines,
            embeddings.tolist()
        ]

        # Insert into Milvus
        print("Inserting into Milvus...")
        self.collection.insert(data)

        # Load collection for search
        self.collection.load()
        print(f"✓ Inserted {len(lines)} documents!")

        return lines

    def search(self, query_text, top_k=3):
        """Search for similar documents"""
        print(f"\n🔍 Searching for: '{query_text}'")

        # Generate query embedding
        query_embedding = self.model.encode([query_text])

        # Define search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        # Perform search
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        # Display results
        print(f"\n📊 Top {top_k} Results:")
        print("-" * 80)

        for i, hits in enumerate(results):
            for j, hit in enumerate(hits):
                distance = hit.distance
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                text = hit.entity.get('text')
                print(f"\n{j+1}. Similarity: {similarity_score:.4f} (Distance: {distance:.4f})")
                print(f"   Text: {text}")

        print("-" * 80)
        return results

    def get_stats(self):
        """Get collection statistics"""
        self.collection.flush()
        num_entities = self.collection.num_entities
        print(f"\n📈 Collection Stats:")
        print(f"   Total documents: {num_entities}")

    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("\n✓ Disconnected from Milvus")


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("MILVUS VECTOR DATABASE DEMO")
    print("=" * 80)

    # Initialize
    db = MilvusVectorDB()

    try:
        # Connect to Milvus
        db.connect()

        # Create collection
        db.create_collection()

        # Create index
        db.create_index()

        # Process sample file
        documents = db.process_file("sample_data.txt")

        # Show stats
        db.get_stats()

        # Perform searches
        print("\n" + "=" * 80)
        print("SIMILARITY SEARCH DEMO")
        print("=" * 80)

        # Search 1
        db.search("How do I learn programming?", top_k=3)

        # Search 2
        db.search("Tell me about machine learning", top_k=3)

        # Search 3
        db.search("What about space exploration?", top_k=3)

        print("\n" + "=" * 80)
        print("✓ Demo completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
