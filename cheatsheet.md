# OpenAI Python SDK Cheatsheet

## Client Setup

```python
from openai import OpenAI, AsyncOpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # or reads env automatically
    max_retries=3,        # default 2
    timeout=30.0,         # default 600s — always override
)

# Async
aclient = AsyncOpenAI()
```

## Responses API (Primary)

```python
response = client.responses.create(
    model="gpt-4o",
    input="Hello",
    instructions="You are helpful.",  # system prompt
)
response.output_text    # str: the text reply
response.status         # "completed" | "incomplete" | "in_progress"
response.id             # str: use as previous_response_id

# Stateful multi-turn
response2 = client.responses.create(
    model="gpt-4o",
    input="Follow-up",
    previous_response_id=response.id,  # chains conversation
)
```

## Chat Completions

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ],
    max_tokens=500,
    temperature=0.7,
)
response.choices[0].message.content  # str
response.choices[0].finish_reason    # "stop" | "tool_calls" | "length"
response.usage.prompt_tokens         # int
response.usage.completion_tokens     # int
```

## Tool Use Loop

```python
TOOLS = [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}, "strict": True}}]

while True:
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    choice = resp.choices[0]
    msg = choice.message
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": [...]})

    if choice.finish_reason == "stop":
        break  # ← terminate here
    for tc in (msg.tool_calls or []):
        args = json.loads(tc.function.arguments)
        result = my_tool(**args)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

## Structured Output (JSON mode)

```python
# JSON object
resp = client.chat.completions.create(
    ...,
    response_format={"type": "json_object"},
)
data = json.loads(resp.choices[0].message.content)

# Pydantic (beta)
class MyModel(BaseModel):
    name: str
    score: int

resp = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=MyModel,
)
obj = resp.choices[0].message.parsed  # MyModel instance
```

## Streaming

```python
with client.chat.completions.create(model=MODEL, messages=messages, stream=True) as stream:
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
```

## Embeddings

```python
resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["text1", "text2"],  # batch
)
vectors = [item.embedding for item in resp.data]  # list[list[float]]
# Dimensions: 1536 (small), 3072 (large)
```

## Audio

```python
# Transcription
f = open("audio.mp3", "rb")
result = client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
result.text  # str

# TTS
resp = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello world")
resp.content  # bytes
```

## Images

```python
resp = client.images.generate(model="dall-e-3", prompt="...", size="1024x1024", n=1)
resp.data[0].url  # str (image URL)
```

## Vision

```python
messages = [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
]}]
```

## Files & Batch

```python
# Upload
f = open("data.jsonl", "rb")
file = client.files.create(file=f, purpose="batch")

# Create batch
batch = client.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
batch.status  # "validating" | "in_progress" | "completed" | "failed"

# Poll
batch = client.batches.retrieve(batch.id)
```

## Error Handling

```python
import openai

try:
    response = client.chat.completions.create(...)
except openai.RateLimitError:          # 429 — retry with backoff
    ...
except openai.APIConnectionError:      # network error — retry
    ...
except openai.AuthenticationError:     # 401 — bad API key, do NOT retry
    ...
except openai.BadRequestError as e:    # 400 — fix parameters
    print(e.body)
except openai.APIStatusError as e:     # catch-all for HTTP errors
    print(e.status_code, e.request_id)
```

## Models Reference

| Model | Input $/1M | Output $/1M | Notes |
|-------|-----------|------------|-------|
| gpt-4o | $2.50 | $10.00 | Flagship |
| gpt-4o-mini | $0.15 | $0.60 | Fast & cheap |
| text-embedding-3-small | $0.02 | — | 1536-dim |
| text-embedding-3-large | $0.13 | — | 3072-dim |
| whisper-1 | $0.006/min | — | STT |
| tts-1 | $15.00/1M chars | — | TTS |
| dall-e-3 | $0.04-0.12/image | — | Image gen |

Batch API = 50% discount on all supported models.
