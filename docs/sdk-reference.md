# SDK Reference

## Endpoint Table

| Method | SDK Call | Key Params | Response Fields |
|--------|----------|-----------|----------------|
| Responses create | `client.responses.create()` | `model`, `input`, `instructions`, `previous_response_id` | `.output_text`, `.status`, `.id` |
| Chat create | `client.chat.completions.create()` | `model`, `messages`, `tools`, `stream`, `max_tokens` | `.choices[0].message.content`, `.choices[0].finish_reason`, `.usage` |
| Embeddings | `client.embeddings.create()` | `model`, `input` (str or list) | `.data[0].embedding` (list[float]) |
| Image generate | `client.images.generate()` | `model`, `prompt`, `size`, `quality` | `.data[0].url` |
| Image edit | `client.images.edit()` | `image`, `mask`, `prompt` | `.data[0].url` |
| Transcription | `client.audio.transcriptions.create()` | `model`, `file`, `language`, `response_format` | `.text` |
| Translation | `client.audio.translations.create()` | `model`, `file` | `.text` |
| TTS | `client.audio.speech.create()` | `model`, `voice`, `input` | `.content` (bytes) |
| File upload | `client.files.create()` | `file`, `purpose` | `.id`, `.status` |
| File retrieve | `client.files.retrieve()` | `file_id` | `.id`, `.status`, `.filename` |
| File content | `client.files.content()` | `file_id` | `.content` (bytes) |
| Batch create | `client.batches.create()` | `input_file_id`, `endpoint`, `completion_window` | `.id`, `.status` |
| Batch retrieve | `client.batches.retrieve()` | `batch_id` | `.status`, `.output_file_id`, `.error_file_id` |
| Fine-tune create | `client.fine_tuning.jobs.create()` | `training_file`, `model`, `suffix` | `.id`, `.status` |
| Fine-tune retrieve | `client.fine_tuning.jobs.retrieve()` | `job_id` | `.status`, `.fine_tuned_model` |
| Vector store | `client.vector_stores.create()` | `name` | `.id`, `.status` |

## finish_reason Values

| Value | Meaning |
|-------|---------|
| `"stop"` | Model finished normally — terminate your loop |
| `"tool_calls"` | Model wants to call tools — execute all, then continue |
| `"length"` | Hit `max_tokens` — response truncated |
| `"content_filter"` | Content policy violation |

## Error Classes

| Class | HTTP | Retryable |
|-------|------|-----------|
| `openai.APIConnectionError` | N/A | Yes |
| `openai.APITimeoutError` | N/A | Yes |
| `openai.RateLimitError` | 429 | Yes |
| `openai.InternalServerError` | 500-504 | Yes |
| `openai.AuthenticationError` | 401 | No |
| `openai.PermissionDeniedError` | 403 | No |
| `openai.NotFoundError` | 404 | No |
| `openai.BadRequestError` | 400 | No |

## Mock Equivalents

| Real API response | MockClient equivalent |
|------------------|----------------------|
| `response.choices[0].message.content` | `"Mock response content."` |
| `response.choices[0].finish_reason` | `"stop"` (default) / `"tool_calls"` (when `tool_choice="required"`) |
| `response.usage.prompt_tokens` | `20` |
| `response.usage.completion_tokens` | `30` |
| `response.usage.prompt_tokens_details.cached_tokens` | `0` (first call) / `500` (repeated call with same prefix) |
| `response.usage.completion_tokens_details.reasoning_tokens` | `0` (gpt-4o) / `150` (o1/o3/o4 models) |
| `emb.data[0].embedding` | `[0.1] * 1536` |
| `response.output_text` | `"Mock output text."` |
| `response.status` | `"completed"` |
| `audio_response.text` | `"Mock transcription text."` |
| `speech_response.content` | `b"mock_audio_bytes"` |
| `image_response.data[0].url` | `"https://mock-image.example.com/img.png"` |
| `client.models.list()[0].id` | `"gpt-4o"` (list of 4 models) |
| `client.moderations.create().results[0].flagged` | `False` |
