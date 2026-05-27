package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rogie2325/kazi"
	"github.com/rogie2325/kazi/config"
	"github.com/rogie2325/kazi/registry"
	"github.com/rogie2325/kazi/secrets"
)

func TestRunWithToolsExecutesToolLoop(t *testing.T) {
	reg := registry.New()
	err := reg.Register(registry.ToolDefinition{
		Name:        "add",
		Description: "Add numbers",
		Source:      registry.ToolSourceNative,
		Parameters: []registry.ToolParameter{
			{Name: "a", Type: "number", Required: true},
			{Name: "b", Type: "number", Required: true},
		},
		Handler: func(ctx context.Context, args map[string]any) (any, error) {
			_ = ctx
			return map[string]any{"sum": 5}, nil
		},
	})
	if err != nil {
		t.Fatalf("register tool: %v", err)
	}

	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test" {
			t.Fatalf("missing auth header: %s", got)
		}

		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		switch atomic.AddInt32(&calls, 1) {
		case 1:
			if len(req.Tools) == 0 {
				t.Fatalf("expected tools in request")
			}
			response := map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role":    "assistant",
							"content": "",
							"tool_calls": []any{
								map[string]any{
									"id":   "call_1",
									"type": "function",
									"function": map[string]any{
										"name":      "add",
										"arguments": "{\"a\":2,\"b\":3}",
									},
								},
							},
						},
					},
				},
				"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
			}
			_ = json.NewEncoder(w).Encode(response)
		case 2:
			if len(req.Messages) < 2 {
				t.Fatalf("expected tool response message")
			}
			last := req.Messages[len(req.Messages)-1]
			if last.Role != "tool" {
				t.Fatalf("expected last role tool, got %s", last.Role)
			}
			if !strings.Contains(last.Content, "\"sum\"") {
				t.Fatalf("expected tool output in message, got %s", last.Content)
			}
			response := map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role":    "assistant",
							"content": "Result is 5",
						},
					},
				},
				"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
			}
			_ = json.NewEncoder(w).Encode(response)
		default:
			t.Fatalf("unexpected request count")
		}
	}))
	defer server.Close()

	ref := secrets.FromLiteral("test")
	cfg := config.LLMConfig{
		Provider: config.LLMProviderOpenAI,
		Model:    "gpt-4o",
		APIKey:   &ref,
		BaseURL:  server.URL,
	}

	client, err := New(cfg)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := client.Run(ctx, "add 2 and 3", kazi.RunOptions{MaxToolCalls: 2}, reg)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.Reply != "Result is 5" {
		t.Fatalf("unexpected reply: %s", result.Reply)
	}
}

func TestStreamChatParsesToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"world\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"echo\",\"arguments\":\"{\\\"message\\\":\\\"hi\\\"}\"}}]}}]}\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer server.Close()

	ref := secrets.FromLiteral("test")
	cfg := config.LLMConfig{
		Provider: config.LLMProviderOpenAI,
		Model:    "gpt-4o",
		APIKey:   &ref,
		BaseURL:  server.URL,
	}

	client, err := New(cfg)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	content, calls, err := client.streamChat(ctx, []message{{Role: "user", Content: "hi"}}, []toolDefinition{{Type: "function", Function: toolFunctionDef{Name: "echo", Parameters: map[string]any{"type": "object"}}}}, "", nil)
	if err != nil {
		t.Fatalf("streamChat: %v", err)
	}
	if content != "Hello world" {
		t.Fatalf("unexpected content: %s", content)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "echo" {
		t.Fatalf("unexpected tool name: %s", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments == "" {
		t.Fatalf("expected tool arguments")
	}
}

func TestRunRetriesOnServerError(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "boom"}})
			return
		}
		response := map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"role":    "assistant",
						"content": "ok",
					},
				},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	ref := secrets.FromLiteral("test")
	cfg := config.LLMConfig{
		Provider: config.LLMProviderOpenAI,
		Model:    "gpt-4o",
		APIKey:   &ref,
		BaseURL:  server.URL,
	}

	client, err := New(cfg)
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	client.maxRetries = 1
	client.retryBackoff = 5 * time.Millisecond

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	result, err := client.Run(ctx, "hi", kazi.RunOptions{}, nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.Reply != "ok" {
		t.Fatalf("unexpected reply: %s", result.Reply)
	}
	if atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}
}
