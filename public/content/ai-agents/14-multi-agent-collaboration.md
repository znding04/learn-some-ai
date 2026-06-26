---
title: "Multi-Agent Collaboration"
topic: ai-agents
order: 14
estimatedTime: "30 minutes"
difficulty: advanced
prerequisites:
  - ai-agents-06
summary: "Learn how multiple AI agents communicate, delegate tasks, reach consensus, and coordinate through shared memory architectures and structured protocols."
---
# Multi-Agent Collaboration

## Overview

A single agent can handle many tasks, but complex problems often benefit from decomposition across multiple specialized
agents. Multi-agent systems divide work among agents that communicate, negotiate, and collaborate -- much like teams of
humans. This lesson covers the core communication protocols, delegation patterns, consensus mechanisms, and coordination
architectures that make multi-agent collaboration effective.

---

## Communication Protocols

Agents need a shared language for exchanging information. The simplest approach is structured message passing where each
message has a sender, receiver, performative (intent), and content.

Common performatives borrowed from speech act theory:

- **INFORM**: Share a fact or observation
- **REQUEST**: Ask another agent to perform an action
- **PROPOSE**: Suggest a plan or solution
- **ACCEPT / REJECT**: Respond to a proposal
- **QUERY**: Ask for information

```python
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict
import asyncio

@dataclass
class Message:
    sender: str
    receiver: str
    performative: str  # INFORM, REQUEST, PROPOSE, ACCEPT, REJECT, QUERY
    content: Any
    reply_to: str | None = None

class MessageBus:
    """Simple publish-subscribe message bus for agent communication."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

    async def send(self, message: Message):
        await self._queues[message.receiver].put(message)

    async def receive(self, agent_id: str) -> Message:
        return await self._queues[agent_id].get()

class Agent:
    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self.bus = bus

    async def send(self, receiver: str, performative: str, content: Any):
        msg = Message(sender=self.agent_id, receiver=receiver,
                      performative=performative, content=content)
        await self.bus.send(msg)

    async def listen(self):
        while True:
            msg = await self.bus.receive(self.agent_id)
            await self.handle(msg)

    async def handle(self, msg: Message):
        raise NotImplementedError
```

---

## Task Delegation Patterns

### Manager-Worker Pattern

A manager agent decomposes a complex task into subtasks and assigns them to specialized worker agents. The manager
collects results, detects failures, and may reassign work.

```python
class ManagerAgent(Agent):
    def __init__(self, agent_id: str, bus: MessageBus, workers: list[str]):
        super().__init__(agent_id, bus)
        self.workers = workers
        self.results: dict[str, Any] = {}

    async def delegate(self, task: dict):
        subtasks = self.decompose(task)
        for i, subtask in enumerate(subtasks):
            worker = self.workers[i % len(self.workers)]
            await self.send(worker, "REQUEST", subtask)

    def decompose(self, task: dict) -> list[dict]:
        # Split task into subtasks based on domain
        return task.get("subtasks", [task])

    async def handle(self, msg: Message):
        if msg.performative == "INFORM":
            self.results[msg.sender] = msg.content
```

### Contract Net Protocol

In the contract net protocol, a manager broadcasts a task announcement. Workers evaluate whether they can perform the
task and submit bids. The manager awards the contract to the best bidder.

1. Manager broadcasts **CALL FOR PROPOSALS**
2. Workers evaluate and respond with **PROPOSE** (bid) or **REJECT**
3. Manager selects winner, sends **ACCEPT** to winner, **REJECT** to others
4. Winner executes task, sends **INFORM** with results

### Auction-Based Allocation

When resources are scarce, auction mechanisms allocate tasks efficiently. Each agent bids based on its estimated cost or
capability. The allocation minimizes total cost:

$$\text{allocation}^* = \arg\min_{\mathbf{a}} \sum_{i=1}^{n} c_i(a_i)$$

where $c_i(a_i)$ is the cost for agent $i$ to perform its assigned task $a_i$.

---

## Consensus Mechanisms

When agents must agree on a shared decision (e.g., which plan to execute), consensus protocols ensure agreement despite
differing local information.

### Majority Voting

The simplest consensus: each agent votes, and the majority wins. For $n$ agents with votes $v_i \in \{0, 1\}$:

$$\text{decision} = \begin{cases} 1 & \text{if } \sum_{i=1}^{n} v_i > \frac{n}{2} \\ 0 & \text{otherwise} \end{cases}$$

### Weighted Voting

Agents may have different expertise levels. Assign weight $w_i$ reflecting agent $i$'s reliability:

$$\text{decision} = \begin{cases} 1 & \text{if } \sum_{i=1}^{n} w_i \cdot v_i > \frac{1}{2}\sum_{i=1}^{n} w_i \\ 0 & \text{otherwise} \end{cases}$$

Weights can be updated over time based on each agent's track record accuracy.

### Iterative Refinement

Agents share their reasoning, update beliefs, and re-vote across multiple rounds until convergence. This mirrors the
Delphi method:

$$\text{belief}_i^{(t+1)} = \alpha \cdot \text{belief}_i^{(t)} + (1-\alpha) \cdot \frac{1}{n-1}\sum_{j \neq i} \text{belief}_j^{(t)}$$

where $\alpha$ controls how much an agent trusts its own prior versus the group average.

---

## Shared Memory Architectures

### Blackboard Systems

A blackboard is a shared data structure that all agents can read and write. Agents monitor the blackboard for relevant
changes and contribute partial solutions:

```python
class Blackboard:
    def __init__(self):
        self.state: dict[str, Any] = {}
        self.subscribers: dict[str, list[callable]] = defaultdict(list)

    def write(self, key: str, value: Any, author: str):
        self.state[key] = {"value": value, "author": author}
        for callback in self.subscribers.get(key, []):
            callback(key, value)

    def read(self, key: str) -> Any:
        return self.state.get(key, {}).get("value")

    def subscribe(self, key: str, callback: callable):
        self.subscribers[key].append(callback)
```

Blackboard systems excel when the problem-solving process is opportunistic -- any agent can contribute whenever it has
relevant knowledge.

### Shared Vector Store

For LLM-based agents, a shared vector database serves as collective memory. Agents write observations and retrieved
facts as embeddings; other agents query the store to benefit from previously gathered knowledge.

---

## Coordination Strategies

| Strategy | Best For | Tradeoff |
|----------|----------|----------|
| Centralized (manager) | Well-defined hierarchical tasks | Single point of failure |
| Contract net | Dynamic task allocation | Communication overhead |
| Blackboard | Opportunistic problem solving | Potential write conflicts |
| Auction | Resource-constrained environments | Requires cost estimation |
| Peer-to-peer | Equal, autonomous agents | Harder to guarantee convergence |

---

## Key Takeaways

- Multi-agent systems decompose complex problems across specialized agents
- Structured message protocols (performatives) ensure clear communication intent
- Contract nets and auctions provide market-inspired task allocation
- Consensus mechanisms (voting, weighted, iterative) enable group decisions
- Shared memory (blackboards, vector stores) provides collective knowledge
- Choose coordination strategy based on task structure, agent autonomy, and fault tolerance requirements

## Exercises

1. **Implement a Message Bus**: Extend the `MessageBus` class to support message priority levels and delivery timestamps. Add a method to retrieve undelivered messages older than a configurable threshold.

2. **Manager-Worker Pipeline**: Implement the full manager-worker pipeline using asyncio. Add fault tolerance: if a worker fails to respond within a timeout, reassign the task to another worker. Track success/failure rates for each worker.

3. **Contract Net Simulation**: Simulate a contract net with 5 worker agents and 1 manager. Each worker should bid based on simulated cost estimates. Run 20 rounds and analyze which workers win more contracts and why.

4. **Consensus Visualization**: Implement the iterative refinement algorithm with 4 agents. Each agent starts with a random belief vector. Visualize how beliefs converge over rounds. Experiment with different values of $\alpha$.

## Further Reading

- Dorri, A., et al. (2018). "Multi-agent systems: A survey." *IEEE Access* — comprehensive overview of multi-agent architectures.
- Lesser, V. R. (1999). "Cooperative Multiagent Systems: A Personal Perspective." *AI Magazine* — foundational reading on coordination.
- Stone, P., & Veloso, M. (2000). "Multiagent Systems: A Survey from a Machine Learning Perspective." *Autonomous Robots* — covers task delegation and auctions.
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*, 2nd ed. Wiley — standard textbook on agent communication and coordination.
- OpenAI's Swarm framework: [https://github.com/openai/swarm](https://github.com/openai/swarm) — multi-agent coordination patterns in production.
