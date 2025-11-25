#!/usr/bin/env python3
"""
FinOps Agent with Multi-Collection Vector Search
================================================
Demonstrates how to:
1. Load queries from multiple domains (cost, budget, utilization, tools)
2. Search across multiple collections
3. Merge context from different sources
4. Build rich prompts for LLM calls
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SearchResult:
    """Represents a single search result with metadata"""
    text: str
    score: float
    source: str  # Which collection it came from
    metadata: Dict[str, Any]


@dataclass
class ContextBundle:
    """Rich context bundle for LLM"""
    query: str
    cost_context: List[SearchResult]
    budget_context: List[SearchResult]
    utilization_context: List[SearchResult]
    tools_context: List[SearchResult]

    def to_prompt_context(self) -> str:
        """Format all context for LLM prompt"""
        sections = []

        if self.cost_context:
            sections.append("## Cost Information\n" + "\n".join(
                f"- {r.text} (relevance: {r.score:.2f})"
                for r in self.cost_context
            ))

        if self.budget_context:
            sections.append("## Budget Information\n" + "\n".join(
                f"- {r.text} (relevance: {r.score:.2f})"
                for r in self.budget_context
            ))

        if self.utilization_context:
            sections.append("## Utilization Information\n" + "\n".join(
                f"- {r.text} (relevance: {r.score:.2f})"
                for r in self.utilization_context
            ))

        if self.tools_context:
            sections.append("## Recommended Tools\n" + "\n".join(
                f"- **{r.metadata.get('tool', 'Unknown')}**: {r.text}"
                for r in self.tools_context
            ))

        return "\n\n".join(sections)


# ============================================================================
# FINOPS VECTOR DATABASE
# ============================================================================

class FinOpsVectorDB:
    """
    Multi-collection vector database for FinOps queries.

    Collections:
    - finops_cost: Cost-related queries and knowledge
    - finops_budget: Budget tracking and allocation
    - finops_utilization: Resource utilization and optimization
    - finops_tools: FinOps tools and their capabilities
    """

    # Collection configurations
    COLLECTIONS = {
        "finops_cost": {
            "description": "Cost analysis queries and knowledge",
            "source_dir": "sample_queries/cost"
        },
        "finops_budget": {
            "description": "Budget tracking and allocation",
            "source_dir": "sample_queries/budget"
        },
        "finops_utilization": {
            "description": "Resource utilization and optimization",
            "source_dir": "sample_queries/utilization"
        },
        "finops_tools": {
            "description": "FinOps tools and capabilities",
            "source_dir": "sample_queries/tools"
        }
    }

    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.collections: Dict[str, Collection] = {}

        # Initialize embedding model
        # all-MiniLM-L6-v2: Good balance of speed and quality
        # Dimension: 384, Max sequence: 256 tokens
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384

    def connect(self):
        """Connect to Milvus server"""
        print(f"Connecting to Milvus at {self.host}:{self.port}...")
        connections.connect("default", host=self.host, port=self.port)
        print("Connected!\n")

    def disconnect(self):
        """Disconnect from Milvus"""
        connections.disconnect("default")
        print("Disconnected from Milvus")

    def _create_collection(self, name: str, description: str) -> Collection:
        """Create a single collection with standard schema"""
        if utility.has_collection(name):
            utility.drop_collection(name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),  # JSON string
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]

        schema = CollectionSchema(fields, description=description)
        collection = Collection(name, schema)

        # Create index for similarity search
        # IVF_FLAT: Good for medium datasets, exact results after coarse search
        index_params = {
            "metric_type": "L2",      # Euclidean distance (lower = more similar)
            "index_type": "IVF_FLAT", # Inverted file index
            "params": {"nlist": 128}  # Number of cluster centroids
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        return collection

    def setup_all_collections(self):
        """Create all FinOps collections"""
        print("Setting up collections...")
        for name, config in self.COLLECTIONS.items():
            self.collections[name] = self._create_collection(name, config["description"])
            print(f"  Created: {name}")
        print()

    def load_data_from_directory(self, base_dir: str = "."):
        """Load data from sample_queries directory into collections"""
        print("Loading data into collections...")

        for coll_name, config in self.COLLECTIONS.items():
            source_dir = os.path.join(base_dir, config["source_dir"])
            if not os.path.exists(source_dir):
                print(f"  Skipping {coll_name}: directory not found")
                continue

            texts = []
            metadata_list = []

            # Load all JSON files in the directory
            for filename in os.listdir(source_dir):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(source_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                for item in data:
                    if coll_name == "finops_tools":
                        # Tools have different structure
                        texts.append(f"{item['tool']}: {item['description']}")
                        metadata_list.append(json.dumps({
                            "tool": item["tool"],
                            "capabilities": item.get("capabilities", []),
                            "category": item.get("category", "")
                        }))
                    else:
                        # Query collections
                        texts.append(item["query"])
                        metadata_list.append(json.dumps({
                            "intent": item.get("intent", ""),
                            "keywords": item.get("keywords", [])
                        }))

            if texts:
                # Generate embeddings for all texts
                embeddings = self.model.encode(texts).tolist()

                # Insert into collection
                self.collections[coll_name].insert([texts, metadata_list, embeddings])
                self.collections[coll_name].load()
                print(f"  Loaded {len(texts)} items into {coll_name}")

        print()

    def search_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        Search a single collection.

        Args:
            collection_name: Name of collection to search
            query: Search query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of SearchResult objects
        """
        if collection_name not in self.collections:
            return []

        collection = self.collections[collection_name]
        query_embedding = self.model.encode([query])

        results = collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param={
                "metric_type": "L2",
                "params": {"nprobe": 10}  # Number of clusters to search
            },
            limit=top_k,
            output_fields=["text", "metadata"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + hit.distance)

                if similarity >= score_threshold:
                    metadata = json.loads(hit.entity.get('metadata', '{}'))
                    search_results.append(SearchResult(
                        text=hit.entity.get('text'),
                        score=similarity,
                        source=collection_name,
                        metadata=metadata
                    ))

        return search_results

    def search_all_collections(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> ContextBundle:
        """
        Search across all collections and build rich context.

        This is the key method for RAG - it gathers relevant context
        from multiple knowledge sources to provide comprehensive answers.
        """
        return ContextBundle(
            query=query,
            cost_context=self.search_collection("finops_cost", query, top_k, score_threshold),
            budget_context=self.search_collection("finops_budget", query, top_k, score_threshold),
            utilization_context=self.search_collection("finops_utilization", query, top_k, score_threshold),
            tools_context=self.search_collection("finops_tools", query, top_k, score_threshold)
        )

    def batch_search(
        self,
        queries: List[str],
        collection_name: str,
        top_k: int = 3
    ) -> Dict[str, List[SearchResult]]:
        """
        Batch search multiple queries efficiently.

        Encodes all queries at once (faster than one-by-one).
        """
        if collection_name not in self.collections:
            return {}

        collection = self.collections[collection_name]

        # Batch encode all queries
        query_embeddings = self.model.encode(queries)

        results = collection.search(
            data=query_embeddings.tolist(),
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "metadata"]
        )

        batch_results = {}
        for query_idx, hits in enumerate(results):
            query = queries[query_idx]
            batch_results[query] = []

            for hit in hits:
                similarity = 1 / (1 + hit.distance)
                metadata = json.loads(hit.entity.get('metadata', '{}'))
                batch_results[query].append(SearchResult(
                    text=hit.entity.get('text'),
                    score=similarity,
                    source=collection_name,
                    metadata=metadata
                ))

        return batch_results


# ============================================================================
# FINOPS AGENT
# ============================================================================

class FinOpsAgent:
    """
    Intelligent FinOps agent that uses vector search to retrieve
    relevant context and generates informed responses.
    """

    # System prompt for the FinOps agent
    SYSTEM_PROMPT = """You are a FinOps expert assistant. Your role is to help users
understand and optimize their cloud costs, manage budgets, and improve resource utilization.

When answering questions:
1. Use the provided context to give accurate, specific answers
2. Recommend appropriate tools when relevant
3. Provide actionable insights and recommendations
4. If the context doesn't contain enough information, say so clearly

Always be specific about which cloud provider or service you're discussing."""

    def __init__(self, db: FinOpsVectorDB):
        self.db = db

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query and build rich context for LLM.

        Returns a structured response with:
        - Retrieved context from all relevant collections
        - Formatted prompt ready for LLM
        - Metadata about the search results
        """
        print(f"\n{'='*80}")
        print(f"Processing Query: {user_query}")
        print(f"{'='*80}\n")

        # Step 1: Search all collections
        context = self.db.search_all_collections(
            query=user_query,
            top_k=3,
            score_threshold=0.25
        )

        # Step 2: Display what was found
        self._display_search_results(context)

        # Step 3: Build the LLM prompt
        prompt = self._build_prompt(user_query, context)

        return {
            "query": user_query,
            "context": context,
            "prompt": prompt,
            "stats": {
                "cost_results": len(context.cost_context),
                "budget_results": len(context.budget_context),
                "utilization_results": len(context.utilization_context),
                "tools_results": len(context.tools_context)
            }
        }

    def _display_search_results(self, context: ContextBundle):
        """Display search results for debugging/visibility"""
        print("Retrieved Context:")
        print("-" * 40)

        if context.cost_context:
            print("\n[COST]")
            for r in context.cost_context:
                print(f"  {r.score:.2f} | {r.text[:60]}...")

        if context.budget_context:
            print("\n[BUDGET]")
            for r in context.budget_context:
                print(f"  {r.score:.2f} | {r.text[:60]}...")

        if context.utilization_context:
            print("\n[UTILIZATION]")
            for r in context.utilization_context:
                print(f"  {r.score:.2f} | {r.text[:60]}...")

        if context.tools_context:
            print("\n[TOOLS]")
            for r in context.tools_context:
                print(f"  {r.score:.2f} | {r.metadata.get('tool', 'Unknown')}")
        print()

    def _build_prompt(self, query: str, context: ContextBundle) -> str:
        """Build a rich prompt for the LLM"""
        context_text = context.to_prompt_context()

        prompt = f"""{self.SYSTEM_PROMPT}

---
# Retrieved Context
{context_text if context_text else "No specific context found. Please provide general guidance."}

---
# User Question
{query}

---
# Your Response
Please provide a helpful, accurate response based on the context above."""

        return prompt


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Demonstrate the FinOps agent"""
    print("=" * 80)
    print("FINOPS AGENT - Multi-Collection Vector Search Demo")
    print("=" * 80)
    print()

    # Initialize
    db = FinOpsVectorDB()

    try:
        # Setup
        db.connect()
        db.setup_all_collections()
        db.load_data_from_directory()

        # Create agent
        agent = FinOpsAgent(db)

        # Example queries
        example_queries = [
            "How can I reduce my AWS EC2 costs?",
            "Are we going over budget this quarter?",
            "Which resources are underutilized and wasting money?",
            "What tools can help with Kubernetes cost allocation?",
        ]

        for query in example_queries:
            result = agent.process_query(query)

            print("\n" + "=" * 80)
            print("GENERATED PROMPT FOR LLM:")
            print("=" * 80)
            print(result["prompt"][:1500] + "..." if len(result["prompt"]) > 1500 else result["prompt"])
            print()
            print(f"Stats: {result['stats']}")
            print()
            input("Press Enter for next query...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
