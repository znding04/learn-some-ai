---
title: "Tool Design & API Integration"
level: intermediate
topic: ai-agents
order: 8
---

# Tool Design & API Integration

## Overview

An AI agent without tools is limited to generating text. With tools, it can read files, query databases, call APIs, send emails, and interact with the real world. **Tool design** is the art of wrapping external capabilities into clean, well-documented interfaces that an LLM can reliably invoke.

### REST API Tool Wrapping

Most external services expose REST APIs. Wrapping a REST API as a tool involves three steps:

1. **Define the tool schema**: Describe the tool's name, purpose, and parameters in a JSON schema that the LLM can understand. The schema tells the model what arguments it needs to supply.
2. **Implement the handler**: Write a function that takes the LLM's parameters, constructs an HTTP request, sends it, and returns a formatted response.
3. **Register the tool**: Add the tool to the agent's tool registry so the LLM knows it exists and can call it.

Good tool design follows the **principle of least surprise**: the tool name should describe what it does (`search_weather`, not `api_call_3`), parameters should have clear types and descriptions, and the response should be concise enough for the LLM to process without hitting token limits.

### Authentication Patterns

Real-world APIs require authentication. The three most common patterns are:

- **API Keys**: The simplest approach. A secret key is included in the request header (e.g., `Authorization: Bearer sk-...`). Store keys in environment variables, never in code.
- **OAuth 2.0**: Used by Google, GitHub, Slack, and most enterprise services. The agent obtains a short-lived access token by exchanging a refresh token. The flow involves a token endpoint and periodic renewal.
- **Session tokens / cookies**: Some web services use session-based auth. The agent logs in once and reuses the session cookie for subsequent requests.

Security is critical. The agent's tool implementation should never log credentials, expose them in error messages, or transmit them over unencrypted channels.

### Rate Limiting and Error Handling

Production APIs enforce rate limits. An agent making rapid successive calls can quickly exhaust its quota. Robust tool design must handle this gracefully.

The standard approach is **exponential backoff with jitter**. After a rate-limit error (HTTP 429), wait before retrying:

$$t_{\text{wait}} = \min(t_{\text{base}} \cdot 2^{n} + \text{rand}(0, t_{\text{jitter}}), \; t_{\text{max}})$$

where $n$ is the retry attempt number, $t_{\text{base}}$ is the initial wait time (e.g., 1 second), $t_{\text{jitter}}$ adds randomness to prevent thundering herd, and $t_{\text{max}}$ caps the maximum wait.

Common HTTP errors and how to handle them:

| Status Code | Meaning | Strategy |
|---|---|---|
| 400 | Bad Request | Fix parameters, do not retry |
| 401 | Unauthorized | Refresh token, retry once |
| 403 | Forbidden | Report to user, do not retry |
| 404 | Not Found | Return empty result |
| 429 | Rate Limited | Exponential backoff |
| 500 | Server Error | Retry with backoff, max 3 times |
| 503 | Service Unavailable | Retry with longer backoff |

### Pagination, Filtering, and Batch Operations

APIs rarely return all results at once. **Pagination** splits results across pages. A tool that wraps a paginated API must either:

- Fetch a single page and let the agent request more if needed (simpler, lower token cost)
- Auto-paginate and aggregate results (complete, but risks large payloads)

The first approach is usually better for agents because it keeps each tool response within the LLM's working memory.

**Filtering** should be exposed as tool parameters when the underlying API supports it. Rather than fetching 1,000 records and having the LLM filter them (expensive and error-prone), pass filter parameters to the API and return only matching results.

**Batch operations** combine multiple requests into one. If an agent needs to update 10 records, a batch endpoint is far more efficient than 10 individual calls. The cost savings follow:

$$C_{\text{batch}} = C_{\text{overhead}} + n \cdot C_{\text{per\_item}} \ll n \cdot (C_{\text{overhead}} + C_{\text{per\_item}})$$

where $n$ is the number of items and $C_{\text{overhead}}$ is the fixed cost per request (DNS, TLS handshake, etc.).

## Key Concepts

- **Tool schema**: A JSON specification of a tool's name, description, and parameters that the LLM uses to decide when and how to call it
- **API key management**: Store secrets in environment variables; never hardcode or log them
- **OAuth 2.0 flow**: Token exchange pattern for accessing protected APIs with short-lived credentials
- **Exponential backoff**: Doubling the wait time between retries to handle rate limits gracefully
- **Pagination**: Splitting large result sets across multiple API calls to manage response size
- **Idempotency**: Ensuring repeated tool calls with the same parameters produce the same result (critical for retry logic)

## Code Examples

### Wrapping a Real API as a Tool

```python
import os
import time
import httpx
from typing import Any

# -------------------------------------------------------------------
# Step 1: Define the tool schema (OpenAI function-calling format)
# -------------------------------------------------------------------
WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a city. Returns temperature, conditions, and humidity.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco' or 'London'",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Temperature units: 'metric' for Celsius, 'imperial' for Fahrenheit",
                },
            },
            "required": ["city"],
        },
    },
}


# -------------------------------------------------------------------
# Step 2: Implement the handler with auth, retries, and error handling
# -------------------------------------------------------------------
class WeatherTool:
    """Wraps the OpenWeatherMap API as an agent tool."""

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        # API key from environment variable -- never hardcoded
        self.api_key = os.environ["OPENWEATHER_API_KEY"]
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = httpx.Client(timeout=10.0)

    def _request_with_retry(self, params: dict) -> httpx.Response:
        """Send a GET request with exponential backoff on failure."""
        for attempt in range(self.max_retries):
            response = self.client.get(self.BASE_URL, params=params)

            if response.status_code == 200:
                return response

            if response.status_code == 429 or response.status_code >= 500:
                # Exponential backoff: 1s, 2s, 4s ...
                wait = min(self.base_delay * (2 ** attempt), 30.0)
                time.sleep(wait)
                continue

            # Non-retryable errors: break immediately
            response.raise_for_status()

        raise Exception(f"API request failed after {self.max_retries} retries")

    def execute(self, city: str, units: str = "metric") -> dict[str, Any]:
        """Execute the tool: call the API and return structured results."""
        params = {
            "q": city,
            "units": units,
            "appid": self.api_key,
        }

        response = self._request_with_retry(params)
        data = response.json()

        # Return a concise summary (avoid dumping the full API response)
        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "conditions": data["weather"][0]["description"],
            "units": "C" if units == "metric" else "F",
        }


# -------------------------------------------------------------------
# Step 3: Register and use in an agent loop
# -------------------------------------------------------------------
import openai
import json

client = openai.OpenAI()
weather = WeatherTool()

TOOLS = [WEATHER_TOOL_SCHEMA]
TOOL_HANDLERS = {"get_current_weather": weather.execute}

def agent_loop(user_message: str) -> str:
    """A simple agent loop that can call the weather tool."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant with weather access."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=TOOLS
    )
    msg = response.choices[0].message

    # If the model wants to call a tool, execute it
    if msg.tool_calls:
        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            result = TOOL_HANDLERS[fn_name](**fn_args)

            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

        # Get the final response with tool results
        final = client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        return final.choices[0].message.content

    return msg.content


print(agent_loop("What's the weather like in Tokyo right now?"))
```

**Line-by-line explanation:**

- `WEATHER_TOOL_SCHEMA` defines the tool in OpenAI function-calling format. The `parameters` field uses JSON Schema so the model knows what arguments to produce.
- `WeatherTool.__init__` loads the API key from an environment variable and sets up an HTTP client with a 10-second timeout.
- `_request_with_retry` implements exponential backoff. It retries on 429 (rate limit) and 5xx (server error) codes, but raises immediately on 4xx client errors.
- `execute` calls the API and extracts only the fields the agent needs -- keeping the tool response concise reduces token usage.
- `agent_loop` demonstrates the full flow: send a message, check if the model wants to call a tool, execute the tool, feed results back, and get the final answer.

## Math/Formulas (KaTeX)

Expected latency for a tool call with retry:

$$\mathbb{E}[T] = p_{\text{success}} \cdot t_{\text{call}} + (1 - p_{\text{success}}) \cdot \sum_{n=1}^{N} p_n \cdot (t_{\text{call}} + t_{\text{base}} \cdot 2^n)$$

where $p_{\text{success}}$ is the probability of success on the first try, $p_n$ is the probability of success on retry $n$, and $N$ is the max retries.

Token cost of tool responses:

$$C_{\text{tokens}} = \sum_{i=1}^{k} \text{len}(r_i) \cdot c_{\text{input}}$$

where $r_i$ is the $i$-th tool response, $k$ is the number of tool calls, and $c_{\text{input}}$ is the per-token cost for input tokens.

## Diagrams

```
Tool Execution Flow
====================

  User: "Weather in Tokyo?"
       |
       v
  +----+----------+
  | LLM decides   |
  | to call tool   |
  +----+----------+
       |
       v
  +----+----------+     +--------------------+
  | Parse tool    |     | Tool Registry      |
  | call args     |---->| get_current_weather|
  +---------------+     | search_web         |
                        | run_code           |
                        +----+---------------+
                             |
                             v
                   +---------+----------+
                   | WeatherTool.execute |
                   +---------+----------+
                             |
                   +---------v----------+
                   | HTTP GET with      |
                   | auth + retry logic |
                   +---------+----------+
                             |
                   +---------v----------+
                   | Parse + format     |
                   | response           |
                   +---------+----------+
                             |
                             v
                   +---------+----------+
                   | Return to LLM      |
                   | as tool result      |
                   +--------------------+


Retry with Exponential Backoff
================================

  Attempt 1 ----X (429 error)
    wait 1s
  Attempt 2 ----X (503 error)
    wait 2s
  Attempt 3 ----OK (200)
    return result
```

## Exercises

1. **Wrap a new API**: Choose a free API (e.g., Open Library, PokeAPI, or JSONPlaceholder). Write a tool schema and handler class with authentication and retry logic.

2. **Pagination**: Modify the tool to handle paginated responses. Implement a parameter `page` that lets the agent request specific pages of results.

3. **Batch tool**: Create a tool that accepts a list of cities and returns weather for all of them in a single call. Compare the token cost vs. calling the single-city tool multiple times.

4. **Rate limit simulation**: Write a mock API that returns 429 on 50% of requests. Verify that your retry logic handles it correctly and measure the average number of retries.

5. **Security audit**: Review the code example and identify three potential security issues. Implement fixes for each one.

## Further Reading

- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [httpx Documentation (Python HTTP client)](https://www.python-httpx.org/)
- [REST API Design Best Practices](https://restfulapi.net/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
