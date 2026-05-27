# Kazi (Go)

Go-first rewrite of Kazi focused on cloud APIs, CLI tooling, concurrency-heavy
orchestration, integrations, and RAG/data workflows.

Status: in progress.
Created by Elijah Rose.

## Quick start

Run the server with OpenAI configured:

```sh
make go-serve ARGS="--addr :8080 --api-key YOUR_API_KEY"
```

Or load from a config file:

```sh
make go-serve ARGS="--config ./kazi.yaml --addr :8080"
```

## Config example

```yaml
llm:
	provider: openai
	model: gpt-4o
	api_key: sk-your-key
```

## HTTP API

- `POST /run`       Single turn response
- `POST /stream`    SSE token stream
- `POST /events`    SSE event stream (token/tool_start/tool_end/done/error)
- `POST /ingest`    Document ingestion (stub)
- `GET  /health`    Health status
- `GET  /metrics`   Metrics and server info
- `GET  /prometheus` Prometheus scrape (auth required)

### Run

```sh
curl -sS http://localhost:8080/run \
	-H "Authorization: Bearer YOUR_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"message":"hello"}'
```

### Stream tokens

```sh
curl -N http://localhost:8080/stream \
	-H "Authorization: Bearer YOUR_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"message":"tell me a joke"}'
```

### Stream events with tools

```sh
curl -N http://localhost:8080/events \
	-H "Authorization: Bearer YOUR_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"message":"Add 2 and 3", "max_tool_calls": 2}'
```

## Example tools

The server registers example tools by default:
- `echo(message: string)`
- `add(a: number, b: number)`

Disable them with:

```sh
make go-serve ARGS="--example-tools=false"
```

## Notes

- `/stream` supports tool calls but only emits tokens.
- `/events` emits tool events for tool-call execution.
- Use `--prometheus=false` to disable the Prometheus endpoint.
- Use `--log-requests=false` to disable structured request logs.
- Use `--log-sample-rate=0.1` to sample 10% of request logs.

