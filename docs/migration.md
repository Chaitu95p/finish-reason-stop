# Migration Guide

## openai SDK v0 → v1

The v1 SDK (released November 2023) is a complete rewrite. Key breaking changes:

### Client Construction

```python
# v0 (deprecated)
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
response = openai.ChatCompletion.create(...)

# v1 (current)
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY automatically
response = client.chat.completions.create(...)
```

### Error Imports

```python
# v0 (deprecated)
from openai.error import RateLimitError, OpenAIError

# v1 (current)
from openai import RateLimitError, APIError
```

### Response Access

```python
# v0 (dict-style, deprecated)
content = response["choices"][0]["message"]["content"]

# v1 (attribute-style)
content = response.choices[0].message.content
```

### Streaming

```python
# v0 (deprecated)
for chunk in openai.ChatCompletion.create(..., stream=True):
    delta = chunk["choices"][0]["delta"].get("content", "")

# v1 (current)
with client.chat.completions.create(..., stream=True) as stream:
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
```

## Chat Completions → Responses API

The Responses API is OpenAI's primary stateful conversation API. Chat Completions still works but is secondary.

| Chat Completions | Responses API |
|-----------------|---------------|
| `client.chat.completions.create()` | `client.responses.create()` |
| `messages=[{"role":"system","content":"..."}]` | `instructions="..."` |
| `messages=[{"role":"user","content":"..."}]` | `input="..."` |
| `choices[0].message.content` | `response.output_text` |
| `choices[0].finish_reason == "stop"` | `response.status == "completed"` |
| Manual messages list for multi-turn | `previous_response_id=response.id` |

### Migration Example

```python
# Chat Completions (v1, still valid)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Hello"},
    ],
)
print(response.choices[0].message.content)
if response.choices[0].finish_reason == "stop":
    print("Done")

# Responses API (preferred)
response = client.responses.create(
    model="gpt-4o",
    instructions="You are helpful.",
    input="Hello",
)
print(response.output_text)
if response.status == "completed":
    print("Done")
```

## Bulk Migration Commands

```bash
# Find v0 module-level calls
grep -r "openai\.ChatCompletion" src/
grep -r "openai\.Embedding" src/
grep -r "from openai.error import" src/
grep -r "openai\.api_key" src/

# Find dict-style response access
grep -r "\['choices'\]" src/
grep -r "\['message'\]" src/
```
