# Agentic AI Complete Guide

## Table of Contents
1. [Agentic AI Architecture Overview](#agentic-ai-architecture-overview)
2. [Reinforcement Learning for Tool Selection](#reinforcement-learning-for-tool-selection)
3. [Memory Systems in Agents](#memory-systems-in-agents)
4. [Session Management](#session-management)
5. [Storage Selection Matrix](#storage-selection-matrix)
6. [Context Management](#context-management)
7. [Complete Agentic Stack Matrix](#complete-agentic-stack-matrix)
8. [FinOps Agent Architecture](#finops-agent-architecture)

---

## Agentic AI Architecture Overview

### What Makes an Agent "Agentic"?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL LLM vs AGENTIC AI                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRADITIONAL LLM (Reactive)          │  AGENTIC AI (Autonomous)             │
│  ───────────────────────────         │  ──────────────────────────          │
│                                      │                                       │
│  User → LLM → Response               │  User → Agent ─┬→ Plan               │
│         │                            │                ├→ Execute             │
│         └─ Single turn               │                ├→ Observe             │
│                                      │                ├→ Reflect             │
│                                      │                └→ Iterate             │
│                                      │                                       │
│  • No memory                         │  • Persistent memory                  │
│  • No tools                          │  • Tool usage                         │
│  • No planning                       │  • Multi-step planning               │
│  • No learning                       │  • Can learn from feedback           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components of an Agent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENT CORE COMPONENTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌─────────────────┐                                │
│                          │   USER INPUT    │                                │
│                          └────────┬────────┘                                │
│                                   │                                          │
│                                   ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           AGENT BRAIN                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │                          LLM CORE                                 │  │ │
│  │  │                    (Claude, GPT-4, etc.)                         │  │ │
│  │  │                                                                   │  │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │  │ │
│  │  │  │  PLANNING   │ │  REASONING  │ │  REFLECTION │ │  LEARNING  │ │  │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   │                                          │
│         ┌─────────────────────────┼─────────────────────────┐               │
│         │                         │                         │               │
│         ▼                         ▼                         ▼               │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐         │
│  │   MEMORY    │          │    TOOLS    │          │   CONTEXT   │         │
│  │             │          │             │          │             │         │
│  │ • Short-term│          │ • APIs      │          │ • RAG       │         │
│  │ • Long-term │          │ • Functions │          │ • Knowledge │         │
│  │ • Episodic  │          │ • Databases │          │ • State     │         │
│  │ • Semantic  │          │ • External  │          │ • History   │         │
│  └─────────────┘          └─────────────┘          └─────────────┘         │
│         │                         │                         │               │
│         └─────────────────────────┴─────────────────────────┘               │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                │
│                          │     OUTPUT      │                                │
│                          │  Action/Response│                                │
│                          └─────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Reinforcement Learning for Tool Selection

### How Agents Learn to Choose Tools

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL FOR TOOL SELECTION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  The Problem: Agent has N tools, must choose the best one for each task    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  User: "What's our EC2 spend this month?"                           │   │
│  │                                                                      │   │
│  │  Available Tools:                                                    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │AWS Cost │ │ Slack   │ │ Jira    │ │ Vector  │ │ SQL     │       │   │
│  │  │Explorer │ │ API     │ │ API     │ │ Search  │ │ Query   │       │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │       │           │           │           │           │             │   │
│  │       ▼           ▼           ▼           ▼           ▼             │   │
│  │    Q=0.95      Q=0.1       Q=0.05      Q=0.7       Q=0.3           │   │
│  │    ↑ BEST                                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Q-Value = Expected reward from taking this action (choosing this tool)    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RL Components in Agent Tool Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL FRAMEWORK FOR AGENTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STATE (s):                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Current user query (embedded)                                      │   │
│  │ • Conversation history                                               │   │
│  │ • Previous tool results                                              │   │
│  │ • User preferences/profile                                           │   │
│  │ • Current context (time, session info)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ACTION (a):                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Select tool T from available tools {T1, T2, ..., Tn}              │   │
│  │ • Choose parameters for the tool                                     │   │
│  │ • Decide to chain multiple tools                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  REWARD (r):                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Task completion: +1.0 (success), 0 (failure)                      │   │
│  │ • User feedback: +0.5 (positive), -0.5 (negative)                   │   │
│  │ • Efficiency: -0.1 × (tokens used / 1000)                           │   │
│  │ • Latency: -0.1 × (seconds / 10)                                    │   │
│  │ • Tool error: -0.3                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  POLICY (π):                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ π(a|s) = probability of choosing action a given state s             │   │
│  │                                                                      │   │
│  │ Can be:                                                              │   │
│  │ • Neural network (trained policy)                                    │   │
│  │ • LLM-based (prompt-driven)                                         │   │
│  │ • Hybrid (LLM + learned preferences)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Selection Approaches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 TOOL SELECTION APPROACHES COMPARISON                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. RULE-BASED (Simple)                                                      │
│  ───────────────────────                                                     │
│  if "cost" in query → use_cost_explorer()                                   │
│  if "budget" in query → use_budget_api()                                    │
│                                                                              │
│  ✅ Fast, predictable                                                       │
│  ❌ Brittle, doesn't generalize                                             │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  2. EMBEDDING SIMILARITY (Semantic)                                          │
│  ──────────────────────────────────                                          │
│  query_embedding = embed("What's our AWS spend?")                           │
│  tool_embeddings = {                                                        │
│      "cost_explorer": embed("AWS cost analysis spending bills"),            │
│      "slack": embed("team messaging communication"),                        │
│  }                                                                          │
│  best_tool = argmax(cosine_similarity(query, tools))                        │
│                                                                              │
│  ✅ Generalizes well, semantic understanding                                │
│  ❌ No learning from outcomes                                               │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  3. LLM-BASED (ReAct Pattern)                                               │
│  ───────────────────────────                                                 │
│  Thought: I need to find AWS costs. Cost Explorer has spending data.       │
│  Action: cost_explorer(time_range="this_month")                             │
│  Observation: Total spend: $45,230                                          │
│  Thought: I have the answer.                                                │
│  Answer: Your EC2 spend this month is $45,230                               │
│                                                                              │
│  ✅ Flexible, can reason about tools                                        │
│  ❌ Token expensive, can hallucinate tools                                  │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  4. REINFORCEMENT LEARNING (Learned)                                         │
│  ───────────────────────────────────                                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   Query ──→ State Encoder ──→ Policy Network ──→ Tool Selection     │   │
│  │                                      ↑                               │   │
│  │                                      │                               │   │
│  │                              Reward Signal                           │   │
│  │                           (success/failure)                          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ✅ Learns from experience, optimizes for outcomes                          │
│  ❌ Needs training data, cold start problem                                 │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  5. HYBRID (Recommended for Production)                                      │
│  ──────────────────────────────────────                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   Query ──┬──→ Embedding Similarity ──→ Top-K Candidates            │   │
│  │           │                                    │                     │   │
│  │           │                                    ▼                     │   │
│  │           └──→ LLM Reasoning ─────────→ Final Selection             │   │
│  │                      ↑                         │                     │   │
│  │                      │                         ▼                     │   │
│  │               Learned Priors ←───────── Outcome Feedback            │   │
│  │           (from past successes)                                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ✅ Best of all approaches                                                  │
│  ✅ Efficient (filters then reasons)                                        │
│  ✅ Improves over time                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation: RL-Enhanced Tool Selection

```python
"""
Reinforcement Learning for Tool Selection
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


@dataclass
class ToolAction:
    tool_name: str
    parameters: Dict
    confidence: float


@dataclass
class Experience:
    """Single experience for learning"""
    state: np.ndarray       # Query embedding + context
    action: str             # Tool selected
    reward: float           # Outcome (-1 to 1)
    next_state: np.ndarray  # State after action
    done: bool              # Episode complete?


class ToolSelector:
    """
    Hybrid tool selector combining:
    1. Embedding similarity (fast candidate generation)
    2. Q-learning (learned preferences)
    3. LLM reasoning (final decision)
    """

    def __init__(self, tools: List[Dict], embedding_model):
        self.tools = tools
        self.model = embedding_model

        # Pre-compute tool embeddings
        self.tool_embeddings = {
            tool['name']: self.model.encode(tool['description'])
            for tool in tools
        }

        # Q-table for learned preferences
        # Maps (query_cluster, tool) → expected reward
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Experience replay buffer
        self.experiences: List[Experience] = []

        # Exploration rate (epsilon-greedy)
        self.epsilon = 0.1

        # Learning rate
        self.alpha = 0.1

        # Discount factor
        self.gamma = 0.95

    def get_candidates(self, query: str, top_k: int = 3) -> List[str]:
        """Stage 1: Get top-K candidates by embedding similarity"""
        query_embedding = self.model.encode(query)

        similarities = {}
        for tool_name, tool_embedding in self.tool_embeddings.items():
            sim = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )
            similarities[tool_name] = sim

        # Sort by similarity
        sorted_tools = sorted(similarities.items(), key=lambda x: -x[1])
        return [t[0] for t in sorted_tools[:top_k]]

    def get_q_value(self, state_cluster: int, tool: str) -> float:
        """Get learned Q-value for state-action pair"""
        return self.q_table[state_cluster][tool]

    def cluster_state(self, query_embedding: np.ndarray) -> int:
        """Cluster query into discrete state (simplified)"""
        # In practice, use k-means or learned clustering
        return int(np.argmax(query_embedding[:10]) % 100)

    def select_tool(
        self,
        query: str,
        context: Optional[Dict] = None,
        explore: bool = True
    ) -> ToolAction:
        """
        Select best tool using hybrid approach:
        1. Get embedding-based candidates
        2. Apply learned Q-values
        3. Epsilon-greedy exploration
        """
        # Get query embedding
        query_embedding = self.model.encode(query)
        state_cluster = self.cluster_state(query_embedding)

        # Stage 1: Embedding similarity candidates
        candidates = self.get_candidates(query, top_k=5)

        # Stage 2: Score with Q-values
        scored_candidates = []
        for tool in candidates:
            embedding_score = np.dot(
                query_embedding,
                self.tool_embeddings[tool]
            )
            q_score = self.get_q_value(state_cluster, tool)

            # Combine scores (weighted)
            combined_score = 0.6 * embedding_score + 0.4 * q_score
            scored_candidates.append((tool, combined_score))

        # Sort by combined score
        scored_candidates.sort(key=lambda x: -x[1])

        # Epsilon-greedy exploration
        if explore and np.random.random() < self.epsilon:
            # Explore: random selection from candidates
            selected = np.random.choice([c[0] for c in scored_candidates])
            confidence = 0.5
        else:
            # Exploit: best candidate
            selected = scored_candidates[0][0]
            confidence = scored_candidates[0][1]

        return ToolAction(
            tool_name=selected,
            parameters={},  # Would be filled by parameter selector
            confidence=confidence
        )

    def update(self, experience: Experience):
        """Update Q-values from experience (Q-learning)"""
        state_cluster = self.cluster_state(experience.state)

        # Current Q-value
        current_q = self.q_table[state_cluster][experience.action]

        # Max Q-value for next state
        if experience.done:
            max_next_q = 0
        else:
            next_cluster = self.cluster_state(experience.next_state)
            next_q_values = [
                self.q_table[next_cluster][tool]
                for tool in self.tool_embeddings.keys()
            ]
            max_next_q = max(next_q_values) if next_q_values else 0

        # Q-learning update
        new_q = current_q + self.alpha * (
            experience.reward + self.gamma * max_next_q - current_q
        )

        self.q_table[state_cluster][experience.action] = new_q

    def record_outcome(
        self,
        query: str,
        tool_used: str,
        success: bool,
        user_feedback: Optional[float] = None,
        latency_ms: float = 0
    ):
        """Record outcome for learning"""
        # Calculate reward
        reward = 1.0 if success else -0.5

        if user_feedback is not None:
            reward += user_feedback * 0.3  # User feedback bonus

        reward -= (latency_ms / 10000) * 0.1  # Latency penalty

        # Create experience
        query_embedding = self.model.encode(query)
        experience = Experience(
            state=query_embedding,
            action=tool_used,
            reward=reward,
            next_state=query_embedding,  # Simplified
            done=True
        )

        # Update Q-values
        self.update(experience)
        self.experiences.append(experience)


# Example tool definitions for FinOps
FINOPS_TOOLS = [
    {
        "name": "aws_cost_explorer",
        "description": "Query AWS Cost Explorer for spending data, cost breakdowns, and billing information",
        "capabilities": ["cost", "spending", "bills", "aws"]
    },
    {
        "name": "budget_tracker",
        "description": "Track budget allocations, utilization, and alerts for cloud spending",
        "capabilities": ["budget", "alerts", "threshold", "forecast"]
    },
    {
        "name": "resource_analyzer",
        "description": "Analyze resource utilization, find underutilized instances, rightsizing recommendations",
        "capabilities": ["utilization", "rightsizing", "optimization", "waste"]
    },
    {
        "name": "vector_search",
        "description": "Search knowledge base for FinOps best practices and documentation",
        "capabilities": ["search", "knowledge", "documentation", "best practices"]
    },
    {
        "name": "sql_query",
        "description": "Query structured data from cost databases and data warehouses",
        "capabilities": ["sql", "database", "reports", "analytics"]
    }
]
```

---

## Memory Systems in Agents

### Memory Type Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT MEMORY HIERARCHY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WORKING MEMORY (Immediate)                        │   │
│  │  ─────────────────────────────────────────────                       │   │
│  │  • Current conversation context                                      │   │
│  │  • Active tool results                                               │   │
│  │  • In-flight reasoning                                               │   │
│  │                                                                      │   │
│  │  Storage: LLM context window                                         │   │
│  │  Duration: Single request                                            │   │
│  │  Size: 4K - 200K tokens                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SHORT-TERM MEMORY (Session)                       │   │
│  │  ─────────────────────────────────────────────                       │   │
│  │  • Conversation history                                              │   │
│  │  • Session state                                                     │   │
│  │  • Recent interactions                                               │   │
│  │                                                                      │   │
│  │  Storage: Redis, In-memory cache                                     │   │
│  │  Duration: Session (minutes to hours)                                │   │
│  │  Size: MBs                                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EPISODIC MEMORY (Experiences)                     │   │
│  │  ─────────────────────────────────────────────                       │   │
│  │  • Past conversations                                                │   │
│  │  • Tool usage history                                                │   │
│  │  • Success/failure records                                           │   │
│  │                                                                      │   │
│  │  Storage: PostgreSQL, MongoDB                                        │   │
│  │  Duration: Days to months                                            │   │
│  │  Size: GBs                                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SEMANTIC MEMORY (Knowledge)                       │   │
│  │  ─────────────────────────────────────────────                       │   │
│  │  • Domain knowledge                                                  │   │
│  │  • Learned facts                                                     │   │
│  │  • User preferences                                                  │   │
│  │                                                                      │   │
│  │  Storage: Vector DB (Milvus, Pinecone)                              │   │
│  │  Duration: Permanent                                                 │   │
│  │  Size: GBs to TBs                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PROCEDURAL MEMORY (Skills)                        │   │
│  │  ─────────────────────────────────────────────                       │   │
│  │  • Tool usage patterns                                               │   │
│  │  • Learned workflows                                                 │   │
│  │  • Optimized strategies                                              │   │
│  │                                                                      │   │
│  │  Storage: Model weights, Policy networks                             │   │
│  │  Duration: Permanent (until retrained)                               │   │
│  │  Size: MBs to GBs                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Implementation Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY IMPLEMENTATION MATRIX                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Memory Type   │ What to Store          │ Technology        │ Query Type   │
│  ─────────────┼────────────────────────┼───────────────────┼──────────────│
│  Working      │ Current context        │ LLM context       │ N/A          │
│               │                        │                   │              │
│  Short-term   │ Session data           │ Redis             │ Key-value    │
│               │ Recent messages        │ Memcached         │              │
│               │                        │                   │              │
│  Episodic     │ Conversations          │ PostgreSQL        │ SQL          │
│               │ Events, logs           │ MongoDB           │ Document     │
│               │                        │ TimescaleDB       │ Time-series  │
│               │                        │                   │              │
│  Semantic     │ Knowledge              │ Milvus            │ Vector       │
│               │ Embeddings             │ Pinecone          │ search       │
│               │ Documents              │ Weaviate          │              │
│               │                        │                   │              │
│  Procedural   │ Policies               │ Model files       │ Inference    │
│               │ Learned behaviors      │ ONNX              │              │
│               │                        │                   │              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Session Management

### Session Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SESSION MANAGEMENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │   User Request  │                                 │
│                         │  + Session ID   │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                      SESSION MANAGER                                    ││
│  │  ┌──────────────────────────────────────────────────────────────────┐  ││
│  │  │  1. Load Session                                                  │  ││
│  │  │     session = redis.get(f"session:{session_id}")                 │  ││
│  │  │                                                                   │  ││
│  │  │  2. Hydrate Context                                               │  ││
│  │  │     • Conversation history (last N messages)                     │  ││
│  │  │     • User preferences                                            │  ││
│  │  │     • Active state (ongoing tasks)                               │  ││
│  │  │                                                                   │  ││
│  │  │  3. Execute Agent                                                 │  ││
│  │  │     response = agent.run(query, context=session_context)         │  ││
│  │  │                                                                   │  ││
│  │  │  4. Update Session                                                │  ││
│  │  │     session.messages.append(query, response)                     │  ││
│  │  │     session.updated_at = now()                                   │  ││
│  │  │     redis.setex(session_key, TTL, session)                       │  ││
│  │  └──────────────────────────────────────────────────────────────────┘  ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                  │                                          │
│          ┌───────────────────────┼───────────────────────┐                 │
│          │                       │                       │                 │
│          ▼                       ▼                       ▼                 │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐           │
│  │    REDIS     │       │  POSTGRESQL  │       │   MILVUS     │           │
│  │  (Hot Data)  │       │ (Cold Data)  │       │ (Semantic)   │           │
│  ├──────────────┤       ├──────────────┤       ├──────────────┤           │
│  │ • Session    │       │ • User       │       │ • Knowledge  │           │
│  │   state      │       │   profiles   │       │   base       │           │
│  │ • Recent     │       │ • Full       │       │ • Similar    │           │
│  │   messages   │       │   history    │       │   queries    │           │
│  │ • Cache      │       │ • Analytics  │       │ • Context    │           │
│  │              │       │              │       │   retrieval  │           │
│  │ TTL: 24h     │       │ Permanent    │       │ Permanent    │           │
│  └──────────────┘       └──────────────┘       └──────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Session Data Model

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import json


@dataclass
class Message:
    role: str           # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class Session:
    """Complete session state"""

    # Identity
    session_id: str
    user_id: str

    # Timestamps
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    # Conversation
    messages: List[Message] = field(default_factory=list)

    # State
    active_task: Optional[str] = None
    tool_results: Dict = field(default_factory=dict)

    # User context
    preferences: Dict = field(default_factory=dict)

    # Metrics
    total_tokens: int = 0
    tool_calls: int = 0

    def to_context(self, max_messages: int = 10) -> str:
        """Convert session to LLM context"""
        recent = self.messages[-max_messages:]
        return "\n".join([
            f"{m.role}: {m.content}"
            for m in recent
        ])

    def to_json(self) -> str:
        """Serialize for Redis storage"""
        return json.dumps({
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata
                }
                for m in self.messages
            ],
            "active_task": self.active_task,
            "tool_results": self.tool_results,
            "preferences": self.preferences,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls
        })

    @classmethod
    def from_json(cls, data: str) -> "Session":
        """Deserialize from Redis storage"""
        d = json.loads(data)
        return cls(
            session_id=d["session_id"],
            user_id=d["user_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            expires_at=datetime.fromisoformat(d["expires_at"]),
            messages=[
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    metadata=m.get("metadata", {})
                )
                for m in d["messages"]
            ],
            active_task=d.get("active_task"),
            tool_results=d.get("tool_results", {}),
            preferences=d.get("preferences", {}),
            total_tokens=d.get("total_tokens", 0),
            tool_calls=d.get("tool_calls", 0)
        )
```

---

## Storage Selection Matrix

### When to Use What

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STORAGE SELECTION DECISION TREE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  What are you storing?                                                       │
│         │                                                                    │
│         ├─► Embeddings / Vectors ──────────────────► MILVUS / PINECONE      │
│         │   (similarity search)                                              │
│         │                                                                    │
│         ├─► Session state (hot) ───────────────────► REDIS                  │
│         │   (fast access, TTL)                                               │
│         │                                                                    │
│         ├─► Structured data ───────────────────────► POSTGRESQL             │
│         │   (users, transactions, reports)                                   │
│         │                                                                    │
│         ├─► Documents / Unstructured ──────────────► MONGODB                │
│         │   (flexible schema)                                                │
│         │                                                                    │
│         ├─► Time-series metrics ───────────────────► TIMESCALEDB            │
│         │   (cost over time)                                                 │
│         │                                                                    │
│         ├─► Full-text + Vector ────────────────────► ELASTICSEARCH          │
│         │   (hybrid search)                          OPENSEARCH              │
│         │                                                                    │
│         └─► Graph relationships ───────────────────► NEO4J                  │
│             (dependencies, org structure)                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Comparison Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STORAGE COMPARISON FOR AGENTS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│              │ PostgreSQL │ Redis    │ Milvus   │ MongoDB  │ Elastic       │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│  PRIMARY USE │ Structured │ Cache    │ Vectors  │ Documents│ Search        │
│              │ data       │ Sessions │          │          │               │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│  QUERY       │ SQL        │ Key-Val  │ ANN      │ JSON     │ DSL + Vector  │
│  LANGUAGE    │            │ Commands │ Search   │ Query    │               │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│  LATENCY     │ 1-10ms     │ <1ms     │ 1-50ms   │ 1-10ms   │ 5-50ms        │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│  SCALE       │ TBs        │ 100s GB  │ Billions │ TBs      │ PBs           │
│              │            │          │ vectors  │          │               │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│  ACID        │ ✅ Full    │ ❌       │ ❌       │ Partial  │ ❌            │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│  VECTOR      │ pgvector   │ Redis    │ ✅       │ Atlas    │ ✅            │
│  SUPPORT     │ extension  │ VSS      │ Native   │ Vector   │ Dense vector  │
│  ────────────┼────────────┼──────────┼──────────┼──────────┼────────────── │
│                                                                              │
│  USE IN AGENTS:                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  PostgreSQL:                                                                 │
│  • User accounts, preferences                                               │
│  • Conversation history (permanent)                                         │
│  • Cost data, reports                                                       │
│  • Tool execution logs                                                      │
│  • Audit trails                                                             │
│                                                                              │
│  Redis:                                                                      │
│  • Active sessions                                                          │
│  • Rate limiting                                                            │
│  • Semantic cache (recent queries)                                          │
│  • Real-time state                                                          │
│  • Pub/sub for multi-agent                                                  │
│                                                                              │
│  Milvus:                                                                     │
│  • Knowledge base (RAG)                                                     │
│  • Similar query matching                                                   │
│  • User interest profiles                                                   │
│  • Document embeddings                                                      │
│  • Semantic search                                                          │
│                                                                              │
│  MongoDB:                                                                    │
│  • Complex agent state                                                      │
│  • Tool configurations                                                      │
│  • Flexible metadata                                                        │
│  • Event sourcing                                                           │
│                                                                              │
│  Elasticsearch:                                                              │
│  • Full-text + semantic search                                              │
│  • Log analysis                                                             │
│  • Hybrid retrieval                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FinOps Agent: Recommended Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINOPS AGENT RECOMMENDED STACK                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         APPLICATION LAYER                            │   │
│  │                                                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │   │  FastAPI     │    │   Agent      │    │   Workers    │         │   │
│  │   │  (API)       │    │   (Brain)    │    │   (Tasks)    │         │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                        │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│  │   REDIS     │     │ POSTGRESQL  │     │   MILVUS    │                  │
│  │             │     │             │     │             │                  │
│  │ • Sessions  │     │ • Users     │     │ • Knowledge │                  │
│  │ • Cache     │     │ • History   │     │ • Queries   │                  │
│  │ • State     │     │ • Cost data │     │ • Tools     │                  │
│  │             │     │ • Audit     │     │             │                  │
│  │ TTL: 24h    │     │ Permanent   │     │ Permanent   │                  │
│  └─────────────┘     └─────────────┘     └─────────────┘                  │
│                                                                              │
│  WHY THIS COMBINATION:                                                      │
│  ────────────────────                                                       │
│  • Redis: Sub-millisecond session access                                   │
│  • PostgreSQL: ACID for financial data, SQL for complex queries           │
│  • Milvus: Optimized vector search for RAG                                 │
│                                                                              │
│  ANTI-PATTERNS TO AVOID:                                                    │
│  ──────────────────────                                                     │
│  ❌ Storing sessions in PostgreSQL (too slow for hot path)                 │
│  ❌ Storing structured data in Milvus (not designed for it)               │
│  ❌ Using Redis for permanent storage (data loss risk)                     │
│  ❌ Vector search in PostgreSQL pgvector at scale (slower than Milvus)    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Context Management

### Context Building Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT BUILDING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Query: "Why did our costs spike last week?"                           │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: RETRIEVE MULTI-SOURCE CONTEXT                             │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │   Session   │  │   Vector    │  │  Structured │  │   Tools    │ │   │
│  │  │   Memory    │  │   Search    │  │   Query     │  │   Results  │ │   │
│  │  │   (Redis)   │  │   (Milvus)  │  │   (Postgres)│  │            │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬─────┘ │   │
│  │         │                │                │                │       │   │
│  │         ▼                ▼                ▼                ▼       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Raw Context Pool                                           │   │   │
│  │  │  • Last 5 messages                                          │   │   │
│  │  │  • 3 similar past queries                                   │   │   │
│  │  │  • Cost data from last 2 weeks                              │   │   │
│  │  │  • Cost anomaly tool results                                │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: RANK & FILTER                                             │   │
│  │                                                                      │   │
│  │  Score each context piece by relevance:                             │   │
│  │  • Embedding similarity to query                                    │   │
│  │  • Recency (newer = higher weight)                                  │   │
│  │  • Source reliability                                               │   │
│  │  • Token budget constraints                                         │   │
│  │                                                                      │   │
│  │  Filter: Remove score < 0.3, deduplicate                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: COMPRESS & FORMAT                                         │   │
│  │                                                                      │   │
│  │  Token Budget: 4000 tokens for context                              │   │
│  │                                                                      │   │
│  │  Compression strategies:                                            │   │
│  │  • Summarize long documents                                         │   │
│  │  • Extract key sentences                                            │   │
│  │  • Remove redundant information                                     │   │
│  │                                                                      │   │
│  │  Format for LLM:                                                    │   │
│  │  <context>                                                          │   │
│  │    <session_history>...</session_history>                          │   │
│  │    <relevant_knowledge>...</relevant_knowledge>                    │   │
│  │    <tool_results>...</tool_results>                                │   │
│  │  </context>                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 4: INJECT INTO PROMPT                                        │   │
│  │                                                                      │   │
│  │  System: You are a FinOps expert...                                │   │
│  │                                                                      │   │
│  │  {formatted_context}                                                │   │
│  │                                                                      │   │
│  │  User: Why did our costs spike last week?                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Context Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT TYPES IN AGENTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TYPE              │ SOURCE           │ USE CASE                            │
│  ─────────────────┼──────────────────┼─────────────────────────────────────│
│                    │                  │                                     │
│  CONVERSATION     │ Session memory   │ Maintain coherent dialogue          │
│  CONTEXT          │ (Redis)          │ Reference previous messages         │
│                    │                  │ Track user intent                   │
│                    │                  │                                     │
│  ─────────────────┼──────────────────┼─────────────────────────────────────│
│                    │                  │                                     │
│  KNOWLEDGE        │ Vector DB        │ Domain expertise                    │
│  CONTEXT          │ (Milvus)         │ Answer factual questions            │
│                    │                  │ Best practices                      │
│                    │                  │                                     │
│  ─────────────────┼──────────────────┼─────────────────────────────────────│
│                    │                  │                                     │
│  DATA             │ Structured DB    │ Real-time metrics                   │
│  CONTEXT          │ (PostgreSQL)     │ Current state                       │
│                    │ APIs             │ Factual answers                     │
│                    │                  │                                     │
│  ─────────────────┼──────────────────┼─────────────────────────────────────│
│                    │                  │                                     │
│  USER             │ User profile     │ Personalization                     │
│  CONTEXT          │ Preferences      │ Role-based responses                │
│                    │                  │ Access control                      │
│                    │                  │                                     │
│  ─────────────────┼──────────────────┼─────────────────────────────────────│
│                    │                  │                                     │
│  TEMPORAL         │ Time service     │ "This week" interpretation         │
│  CONTEXT          │ Calendar         │ Deadline awareness                  │
│                    │                  │ Seasonality                         │
│                    │                  │                                     │
│  ─────────────────┼──────────────────┼─────────────────────────────────────│
│                    │                  │                                     │
│  TOOL             │ Previous tool    │ Build on prior results              │
│  CONTEXT          │ executions       │ Chain reasoning                     │
│                    │                  │ Avoid redundant calls               │
│                    │                  │                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Agentic Stack Matrix

### What You Might Be Missing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENTIC AI COMPLETENESS CHECKLIST                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  COMPONENT          │ HAVE? │ TECHNOLOGY           │ PRIORITY │ COMPLEXITY │
│  ──────────────────┼───────┼──────────────────────┼──────────┼────────────│
│                     │       │                      │          │            │
│  CORE               │       │                      │          │            │
│  ├─ LLM Brain       │ [ ]   │ Claude/GPT-4         │ CRITICAL │ Low        │
│  ├─ Tool Calling    │ [ ]   │ Function calling     │ CRITICAL │ Medium     │
│  └─ Prompt Mgmt     │ [ ]   │ Templates/Chains     │ HIGH     │ Low        │
│                     │       │                      │          │            │
│  MEMORY             │       │                      │          │            │
│  ├─ Working Memory  │ [ ]   │ LLM context          │ CRITICAL │ Low        │
│  ├─ Session Store   │ [ ]   │ Redis                │ HIGH     │ Low        │
│  ├─ Episodic Store  │ [ ]   │ PostgreSQL           │ MEDIUM   │ Medium     │
│  └─ Semantic Store  │ [ ]   │ Milvus               │ HIGH     │ Medium     │
│                     │       │                      │          │            │
│  RETRIEVAL          │       │                      │          │            │
│  ├─ Vector Search   │ [ ]   │ Milvus/Pinecone      │ HIGH     │ Medium     │
│  ├─ Hybrid Search   │ [ ]   │ BM25 + Vector        │ MEDIUM   │ High       │
│  ├─ Reranking       │ [ ]   │ Cohere/Cross-encoder │ LOW      │ Medium     │
│  └─ Query Routing   │ [ ]   │ Classifier/LLM       │ MEDIUM   │ Medium     │
│                     │       │                      │          │            │
│  LEARNING           │       │                      │          │            │
│  ├─ Feedback Loop   │ [ ]   │ User ratings         │ MEDIUM   │ Low        │
│  ├─ RL Tool Select  │ [ ]   │ Q-learning/Bandit    │ LOW      │ High       │
│  └─ Fine-tuning     │ [ ]   │ Model training       │ LOW      │ Very High  │
│                     │       │                      │          │            │
│  ORCHESTRATION      │       │                      │          │            │
│  ├─ Multi-step Plan │ [ ]   │ ReAct/CoT            │ HIGH     │ Medium     │
│  ├─ Error Recovery  │ [ ]   │ Retry/Fallback       │ HIGH     │ Medium     │
│  ├─ Async Execution │ [ ]   │ Celery/Background    │ MEDIUM   │ Medium     │
│  └─ Multi-Agent     │ [ ]   │ Coordinator pattern  │ LOW      │ High       │
│                     │       │                      │          │            │
│  MONITORING         │       │                      │          │            │
│  ├─ Tracing         │ [ ]   │ LangSmith/Arize      │ HIGH     │ Low        │
│  ├─ Logging         │ [ ]   │ Structured logs      │ HIGH     │ Low        │
│  ├─ Metrics         │ [ ]   │ Prometheus           │ MEDIUM   │ Low        │
│  └─ Evaluation      │ [ ]   │ LLM judges/Benchmarks│ MEDIUM   │ Medium     │
│                     │       │                      │          │            │
│  SAFETY             │       │                      │          │            │
│  ├─ Input Validation│ [ ]   │ Schema/Guardrails    │ CRITICAL │ Low        │
│  ├─ Output Filtering│ [ ]   │ Content filter       │ HIGH     │ Low        │
│  ├─ Rate Limiting   │ [ ]   │ Redis/Token bucket   │ HIGH     │ Low        │
│  └─ Access Control  │ [ ]   │ RBAC/Policies        │ HIGH     │ Medium     │
│                     │       │                      │          │            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack by Maturity Level

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT MATURITY LEVELS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 1: MVP (Week 1)                                                       │
│  ─────────────────────                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • LLM API (Claude/GPT)                                             │   │
│  │  • Basic prompt template                                            │   │
│  │  • 2-3 hardcoded tools                                              │   │
│  │  • In-memory session (lost on restart)                              │   │
│  │  • No persistence                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LEVEL 2: FUNCTIONAL (Month 1)                                               │
│  ────────────────────────────                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  + Redis for sessions                                               │   │
│  │  + Milvus for knowledge RAG                                         │   │
│  │  + PostgreSQL for history                                           │   │
│  │  + Error handling & retries                                         │   │
│  │  + Basic logging                                                    │   │
│  │  + 5-10 tools                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LEVEL 3: PRODUCTION (Quarter 1)                                             │
│  ──────────────────────────────                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  + Multi-step planning (ReAct)                                      │   │
│  │  + Semantic caching                                                 │   │
│  │  + Feedback collection                                              │   │
│  │  + Tracing & monitoring                                             │   │
│  │  + Rate limiting & auth                                             │   │
│  │  + Async task execution                                             │   │
│  │  + 10-20 tools                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LEVEL 4: ADVANCED (Year 1)                                                  │
│  ─────────────────────────                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  + RL-based tool selection                                          │   │
│  │  + Hybrid search (BM25 + vector)                                    │   │
│  │  + Reranking models                                                 │   │
│  │  + A/B testing framework                                            │   │
│  │  + Custom model fine-tuning                                         │   │
│  │  + Multi-agent collaboration                                        │   │
│  │  + Self-improvement loops                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FinOps Agent Architecture

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINOPS AGENT - COMPLETE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌──────────────┐                               │
│                              │    Users     │                               │
│                              │  Slack/Web   │                               │
│                              └──────┬───────┘                               │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          API LAYER                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │   FastAPI   │  │    Auth     │  │    Rate     │                  │   │
│  │  │   Gateway   │  │   (JWT)     │  │   Limiter   │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       SESSION MANAGER                                │   │
│  │                                                                      │   │
│  │   Load Session ──► Hydrate Context ──► Execute ──► Save Session    │   │
│  │         │                                               │            │   │
│  │         ▼                                               ▼            │   │
│  │  ┌─────────────┐                                ┌─────────────┐     │   │
│  │  │    REDIS    │                                │    REDIS    │     │   │
│  │  │  (Session)  │                                │  (Updated)  │     │   │
│  │  └─────────────┘                                └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT CORE                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                      CONTEXT BUILDER                          │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │   │
│  │  │  │ Session │  │ Vector  │  │  SQL    │  │ User    │         │  │   │
│  │  │  │ History │  │ Search  │  │ Query   │  │ Profile │         │  │   │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │  │   │
│  │  │       └────────────┴───────────┴────────────┘               │  │   │
│  │  │                         │                                    │  │   │
│  │  │                         ▼                                    │  │   │
│  │  │              ┌─────────────────────┐                        │  │   │
│  │  │              │   Merged Context    │                        │  │   │
│  │  │              └─────────────────────┘                        │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                │                                     │   │
│  │                                ▼                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                       LLM REASONING                           │  │   │
│  │  │                                                               │  │   │
│  │  │   System Prompt + Context + Query ──► Claude API ──► Plan    │  │   │
│  │  │                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                │                                     │   │
│  │                                ▼                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                      TOOL SELECTOR                            │  │   │
│  │  │                                                               │  │   │
│  │  │  Query ──► Embedding Match ──► Q-Value Boost ──► Select Tool │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│  │   │
│  │  │  │AWS Cost │ │ Budget  │ │Resource │ │ Vector  │ │  SQL    ││  │   │
│  │  │  │Explorer │ │ Tracker │ │Analyzer │ │ Search  │ │ Query   ││  │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘│  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                │                                     │   │
│  │                                ▼                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                     TOOL EXECUTOR                             │  │   │
│  │  │                                                               │  │   │
│  │  │   Execute ──► Validate Result ──► Retry if Failed            │  │   │
│  │  │                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                │                                     │   │
│  │                                ▼                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    RESPONSE GENERATOR                         │  │   │
│  │  │                                                               │  │   │
│  │  │   Tool Results + Context ──► LLM ──► Final Response          │  │   │
│  │  │                                                               │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│         ┌───────────────────────────┼───────────────────────────┐           │
│         │                           │                           │           │
│         ▼                           ▼                           ▼           │
│  ┌─────────────┐            ┌─────────────┐            ┌─────────────┐     │
│  │   MILVUS    │            │ POSTGRESQL  │            │    REDIS    │     │
│  │             │            │             │            │             │     │
│  │ Collections:│            │ Tables:     │            │ Keys:       │     │
│  │ • cost      │            │ • users     │            │ • sessions  │     │
│  │ • budget    │            │ • history   │            │ • cache     │     │
│  │ • util      │            │ • cost_data │            │ • rate_limit│     │
│  │ • tools     │            │ • audit_log │            │             │     │
│  └─────────────┘            └─────────────┘            └─────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Summary

```python
# finops_agent_complete.py - Putting it all together

from finops_agent import FinOpsVectorDB, FinOpsAgent  # Vector search
from tool_selector import ToolSelector, FINOPS_TOOLS   # RL tool selection
from session_manager import Session, SessionManager    # Session management


class FinOpsAgentComplete:
    """Complete FinOps agent with all components"""

    def __init__(self):
        # Vector DB for RAG
        self.vector_db = FinOpsVectorDB()
        self.vector_db.connect()

        # Tool selector with RL
        self.tool_selector = ToolSelector(
            tools=FINOPS_TOOLS,
            embedding_model=self.vector_db.model
        )

        # Session manager
        self.session_manager = SessionManager(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://localhost:5432/finops"
        )

        # Base agent
        self.agent = FinOpsAgent(self.vector_db)

    def process(self, user_id: str, session_id: str, query: str) -> str:
        """Process user query with full context"""

        # 1. Load or create session
        session = self.session_manager.get_or_create(session_id, user_id)

        # 2. Build context from multiple sources
        context = self.agent.db.search_all_collections(query)

        # 3. Select best tool using RL
        tool_action = self.tool_selector.select_tool(query)

        # 4. Execute tool
        tool_result = self.execute_tool(tool_action)

        # 5. Generate response
        response = self.generate_response(query, context, tool_result)

        # 6. Update session
        session.messages.append(Message("user", query))
        session.messages.append(Message("assistant", response))
        self.session_manager.save(session)

        # 7. Record outcome for learning
        self.tool_selector.record_outcome(
            query=query,
            tool_used=tool_action.tool_name,
            success=True  # Would be determined by actual outcome
        )

        return response
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC AI QUICK REFERENCE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MEMORY                        │  STORAGE                                   │
│  ──────                        │  ───────                                   │
│  Working   → LLM context       │  Sessions  → Redis                         │
│  Short-term → Redis            │  Users     → PostgreSQL                    │
│  Episodic  → PostgreSQL        │  Knowledge → Milvus                        │
│  Semantic  → Milvus            │  Logs      → PostgreSQL                    │
│                                │                                            │
│  TOOL SELECTION                │  CONTEXT SOURCES                           │
│  ──────────────                │  ───────────────                           │
│  Rule-based  → Simple, brittle │  Session    → Redis                        │
│  Embedding   → Semantic match  │  Knowledge  → Milvus                       │
│  LLM (ReAct) → Flexible        │  Data       → PostgreSQL/APIs             │
│  RL          → Learns          │  User       → Profile store               │
│  Hybrid      → Best (combine)  │                                            │
│                                │                                            │
│  RL COMPONENTS                 │  PRIORITIES                                │
│  ─────────────                 │  ──────────                                │
│  State  → Query + context      │  1. LLM + Tools (core)                    │
│  Action → Tool selection       │  2. Session + RAG (memory)                │
│  Reward → Success + feedback   │  3. Monitoring (observability)            │
│  Policy → Neural net / LLM     │  4. RL + Learning (optimization)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Agentic AI Guide v1.0 - For FinOps Agent Development*
