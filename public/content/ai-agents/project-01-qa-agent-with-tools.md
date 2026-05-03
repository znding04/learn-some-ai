---
title: "Project: Build a Q&A Agent with Tool Use"
level: intermediate
topic: ai-agents
order: 14
---

# Project: Build a Q&A Agent with Tool Use

## Overview

In this project you will build a question-answering agent that can decide **when and which external tool to call** in order to answer a user's question. The agent combines three tools -- a web search API, an arithmetic calculator, and a local knowledge base -- into a single reasoning loop. By the end you will have a working Python system that accepts natural-language questions, plans a sequence of tool calls, executes them, and synthesises a final answer.

The core idea is the **ReAct** (Reason + Act) pattern: the language model produces a *thought*, selects an *action*, observes the *result*, and repeats until it can give a final answer.

## Key Concepts

| Concept | Why It Matters |
|---|---|
| Tool-use prompting | Tells the LLM which tools exist and how to invoke them |
| ReAct loop | Alternates reasoning with action so the agent self-corrects |
| Function dispatch | Maps the model's chosen tool name to real Python functions |
| Token budget | Each loop iteration costs tokens; a maximum iteration cap prevents runaway spending |

The probability that the agent picks the correct tool on the first try depends on how well the tool descriptions match the query. Formally, if we model tool selection as a softmax over $k$ tools with logits $z_i$, the probability of choosing tool $j$ is:

$$P(\text{tool}_j) = \frac{e^{z_j}}{\sum_{i=1}^{k} e^{z_i}}$$

Better descriptions sharpen the logit for the right tool and push this probability toward 1.

## Code Examples

### 1. Define the tools

```python
import math
import json
import requests
from typing import Any

# --- Tool implementations ---

def web_search(query: str) -> str:
    """Search the web via a simple API and return the top 3 snippets."""
    # Replace with your preferred search API (SerpAPI, Tavily, etc.)
    resp = requests.get(
        "https://api.tavily.com/search",
        params={"query": query, "max_results": 3},
        headers={"Authorization": "Bearer YOUR_API_KEY"},
    )
    results = resp.json().get("results", [])
    return "\n".join(r["content"] for r in results)


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return "Error: invalid characters in expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def knowledge_base(topic: str) -> str:
    """Look up a topic in our local JSON knowledge base."""
    with open("kb.json") as f:
        kb = json.load(f)
    entry = kb.get(topic.lower())
    if entry:
        return entry
    return "No entry found for that topic."


TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {
        "fn": web_search,
        "description": "Search the internet for current information.",
        "parameters": {"query": "string"},
    },
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a math expression (e.g. '2+2', '1024/8').",
        "parameters": {"expression": "string"},
    },
    "knowledge_base": {
        "fn": knowledge_base,
        "description": "Look up a topic in the local knowledge base.",
        "parameters": {"topic": "string"},
    },
}
```

### 2. Build the ReAct loop

```python
import openai  # works with any OpenAI-compatible API

client = openai.OpenAI()

SYSTEM_PROMPT = """You are a helpful Q&A agent. You have access to these tools:

{tool_descriptions}

On each turn, respond with EXACTLY one JSON object:
  {{"thought": "...", "action": "tool_name", "input": "..."}}
When you have the final answer, respond with:
  {{"thought": "...", "answer": "..."}}
""".format(
    tool_descriptions="\n".join(
        f"- {name}: {t['description']}" for name, t in TOOLS.items()
    )
)

MAX_ITERATIONS = 6


def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": text})

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "Respond with valid JSON."})
            continue

        # Check for final answer
        if "answer" in parsed:
            return parsed["answer"]

        # Dispatch tool
        action = parsed.get("action")
        tool_input = parsed.get("input", "")
        if action not in TOOLS:
            messages.append(
                {"role": "user", "content": f"Unknown tool '{action}'. Try again."}
            )
            continue

        observation = TOOLS[action]["fn"](tool_input)
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Sorry, I could not find an answer within the iteration limit."
```

### 3. Run it

```python
if __name__ == "__main__":
    q = "What is the population of France divided by 7?"
    print(run_agent(q))
    # Expected flow:
    #   Thought -> web_search("population of France")
    #   Observation -> "The population of France is approximately 68 million..."
    #   Thought -> calculator("68000000 / 7")
    #   Observation -> "9714285.714285714"
    #   Answer -> "Approximately 9,714,286"
```

## Diagrams

```
+--------------------+
|   User Question    |
+--------+-----------+
         |
         v
+--------+-----------+
|   LLM  (Thought)   |<-----------------------+
+--------+-----------+                         |
         |                                     |
    action + input                        Observation
         |                                     |
         v                                     |
+--------+-----------+                         |
|  Tool Dispatcher   |                         |
+--+------+------+---+                         |
   |      |      |                             |
   v      v      v                             |
 Web   Calc    KB                              |
Search                                         |
   |      |      |                             |
   +------+------+-----------------------------+
         result
```

The loop repeats until the model emits an `"answer"` key or hits `MAX_ITERATIONS`.

## Exercises

1. **Add a new tool** -- Implement a `weather(city: str)` tool using a free weather API and register it in `TOOLS`. Verify the agent uses it when asked about weather.
2. **Streaming output** -- Modify `run_agent` to yield intermediate thoughts so the user can watch the reasoning in real time.
3. **Cost tracking** -- After each LLM call, log `response.usage.prompt_tokens` and `response.usage.completion_tokens`. Compute total cost assuming a price of $c$ per 1 M input tokens: $\text{cost} = \frac{n_{\text{input}} \cdot c}{10^6}$.
4. **Retry with back-off** -- Wrap the API call in an exponential back-off loop so transient failures do not crash the agent.
5. **Evaluation harness** -- Create 10 question-answer pairs and measure the agent's accuracy. Report precision as $P = \frac{\text{correct}}{\text{total}}$.

## Further Reading

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
