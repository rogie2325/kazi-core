package serve

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rogie2325/kazi"
	"github.com/rogie2325/kazi/config"
	"github.com/rogie2325/kazi/registry"
)

type stubLLM struct {
	run    kazi.RunResult
	events []kazi.StreamEvent
}

func (s stubLLM) Run(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (kazi.RunResult, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	return s.run, nil
}

func (s stubLLM) Stream(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan string, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	ch := make(chan string, 1)
	go func() {
		defer close(ch)
		ch <- s.run.Reply
	}()
	return ch, nil
}

func (s stubLLM) StreamEvents(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan kazi.StreamEvent, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	ch := make(chan kazi.StreamEvent, len(s.events))
	go func() {
		defer close(ch)
		for _, event := range s.events {
			ch <- event
		}
	}()
	return ch, nil
}

func newTestServer(t *testing.T, opts Options, llm kazi.LLMClient) *httptest.Server {
	t.Helper()
	cfg := config.DefaultConfig()
	k := kazi.New(cfg)
	if llm != nil {
		k.SetLLMClient(llm)
	}
	opts.LogRequests = false
	opts.LogSampleRate = 0
	server := New(k, opts)
	return httptest.NewServer(server.Handler())
}

type testLogger struct {
	mu     sync.Mutex
	infos  []map[string]any
	errors []map[string]any
}

func (t *testLogger) Info(_ string, fields map[string]any) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.infos = append(t.infos, fields)
}

func (t *testLogger) Error(_ string, fields map[string]any) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.errors = append(t.errors, fields)
}

func newServerWithLogger(t *testing.T, opts Options, llm kazi.LLMClient, logger Logger) *httptest.Server {
	t.Helper()
	cfg := config.DefaultConfig()
	k := kazi.New(cfg)
	if llm != nil {
		k.SetLLMClient(llm)
	}
	opts.Logger = logger
	server := New(k, opts)
	return httptest.NewServer(server.Handler())
}

func TestMetricsRequiresAuth(t *testing.T) {
	server := newTestServer(t, Options{APIKey: "secret"}, nil)
	defer server.Close()

	resp, err := http.Get(server.URL + "/metrics")
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", resp.StatusCode)
	}

	req, _ := http.NewRequest(http.MethodGet, server.URL+"/metrics", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("auth request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
}

func TestPrometheusRequiresAuth(t *testing.T) {
	server := newTestServer(t, Options{APIKey: "secret", EnablePrometheus: true}, nil)
	defer server.Close()

	resp, err := http.Get(server.URL + "/prometheus")
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", resp.StatusCode)
	}

	req, _ := http.NewRequest(http.MethodGet, server.URL+"/prometheus", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("auth request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "kazi_http_requests_total") {
		t.Fatalf("expected prometheus metrics")
	}
}

func TestRequestIDSanitized(t *testing.T) {
	server := newTestServer(t, Options{}, nil)
	defer server.Close()

	req, _ := http.NewRequest(http.MethodGet, server.URL+"/health", nil)
	req.Header.Set("X-Request-ID", "bad_id")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	requestID := resp.Header.Get("X-Request-ID")
	if requestID == "" {
		t.Fatalf("expected request id header")
	}
	if requestID == "bad_id" {
		t.Fatalf("expected sanitized request id")
	}
	re := regexp.MustCompile(`^[A-Za-z0-9-]{1,64}$`)
	if !re.MatchString(requestID) {
		t.Fatalf("unexpected request id: %q", requestID)
	}

	req, _ = http.NewRequest(http.MethodGet, server.URL+"/health", nil)
	req.Header.Set("X-Request-ID", "abc-123")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.Header.Get("X-Request-ID") != "abc-123" {
		t.Fatalf("expected request id passthrough")
	}
}

func TestRateLimitExceeded(t *testing.T) {
	server := newTestServer(t, Options{RateLimitPerMinute: 1}, nil)
	defer server.Close()

	resp, err := http.Get(server.URL + "/health")
	if err != nil {
		t.Fatalf("first request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	resp, err = http.Get(server.URL + "/health")
	if err != nil {
		t.Fatalf("second request: %v", err)
	}
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}
}

func TestStreamFiltersToolEvents(t *testing.T) {
	llm := stubLLM{
		events: []kazi.StreamEvent{
			{Type: "token", Data: "hello"},
			{Type: "tool_start", Data: map[string]any{"name": "echo"}},
			{Type: "tool_end", Data: map[string]any{"name": "echo"}},
			{Type: "done", Data: "ok"},
		},
	}
	server := newTestServer(t, Options{}, llm)
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/stream", bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)
	if !strings.Contains(text, "\"token\"") {
		t.Fatalf("expected token payload, got %s", text)
	}
	if strings.Contains(text, "tool_start") {
		t.Fatalf("unexpected tool event in stream: %s", text)
	}
}

func TestStreamTimeoutReturnsError(t *testing.T) {
	server := newTestServer(t, Options{RequestTimeout: 50 * time.Millisecond}, silentLLM{})
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/stream", bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)
	if !strings.Contains(text, "\"error\":\"timeout\"") {
		t.Fatalf("expected timeout error, got %s", text)
	}
}

func TestEventsLogsToolError(t *testing.T) {
	logger := &testLogger{}
	llm := stubLLM{events: []kazi.StreamEvent{
		{Type: "tool_end", Data: map[string]any{"name": "echo", "id": "tool_1", "error": "boom"}},
		{Type: "done", Data: "ok"},
	}}
	server := newServerWithLogger(t, Options{LogRequests: false, LogSampleRate: 0}, llm, logger)
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)
	resp, err := http.Post(server.URL+"/events", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	_, _ = io.ReadAll(resp.Body)

	logger.mu.Lock()
	defer logger.mu.Unlock()
	if len(logger.errors) == 0 {
		t.Fatalf("expected tool error log")
	}
	fields := logger.errors[0]
	if fields["tool_name"] != "echo" {
		t.Fatalf("expected tool_name echo, got %v", fields["tool_name"])
	}
	if fields["error"] != "boom" {
		t.Fatalf("expected error boom, got %v", fields["error"])
	}
}

func TestEventsLogsStreamError(t *testing.T) {
	logger := &testLogger{}
	llm := stubLLM{events: []kazi.StreamEvent{
		{Type: "error", Data: "stream failed"},
	}}
	server := newServerWithLogger(t, Options{LogRequests: false, LogSampleRate: 0}, llm, logger)
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)
	resp, err := http.Post(server.URL+"/events", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	_, _ = io.ReadAll(resp.Body)

	logger.mu.Lock()
	defer logger.mu.Unlock()
	if len(logger.errors) == 0 {
		t.Fatalf("expected stream error log")
	}
	fields := logger.errors[len(logger.errors)-1]
	if fields["route"] != "events" {
		t.Fatalf("expected route events, got %v", fields["route"])
	}
	if fields["error"] != "stream failed" {
		t.Fatalf("expected error stream failed, got %v", fields["error"])
	}
}

func TestMaxConcurrentLimitsRequests(t *testing.T) {
	block := make(chan struct{})
	ready := make(chan struct{})
	llm := blockingLLM{ready: ready, block: block}
	server := newTestServer(t, Options{MaxConcurrent: 1}, llm)
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)

	go func() {
		req, _ := http.NewRequest(http.MethodPost, server.URL+"/run", bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")
		_, _ = http.DefaultClient.Do(req)
	}()

	select {
	case <-ready:
	case <-time.After(2 * time.Second):
		t.Fatalf("timeout waiting for run")
	}

	resp, err := http.Post(server.URL+"/run", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("second request: %v", err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", resp.StatusCode)
	}

	close(block)
}

func TestLogSamplingCapturesRequests(t *testing.T) {
	logger := &testLogger{}
	server := newServerWithLogger(t, Options{LogRequests: true, LogSampleRate: 1.0}, nil, logger)
	defer server.Close()

	resp, err := http.Get(server.URL + "/health")
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	logger.mu.Lock()
	defer logger.mu.Unlock()
	if len(logger.infos) == 0 {
		t.Fatalf("expected request logs")
	}
	if logger.infos[0]["route"] != "health" {
		t.Fatalf("expected route health, got %v", logger.infos[0]["route"])
	}
}

func TestLogSamplingSkipsRequests(t *testing.T) {
	logger := &testLogger{}
	server := newServerWithLogger(t, Options{LogRequests: true, LogSampleRate: 0}, nil, logger)
	defer server.Close()

	resp, err := http.Get(server.URL + "/health")
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	logger.mu.Lock()
	defer logger.mu.Unlock()
	if len(logger.infos) != 0 {
		t.Fatalf("expected no logs, got %d", len(logger.infos))
	}
}

func TestMetricsCountsRequests(t *testing.T) {
	llm := stubLLM{run: kazi.RunResult{Reply: "ok"}}
	server := newTestServer(t, Options{}, llm)
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)
	resp, err := http.Post(server.URL+"/run", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatalf("run request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	resp, err = http.Get(server.URL + "/metrics")
	if err != nil {
		t.Fatalf("metrics request: %v", err)
	}
	var payload map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	counts, ok := payload["requests_total"].(map[string]any)
	if !ok {
		t.Fatalf("expected requests_total map")
	}
	runCountRaw, ok := counts["run"]
	if !ok {
		t.Fatalf("expected run count")
	}
	runCount, ok := runCountRaw.(float64)
	if !ok || runCount < 1 {
		t.Fatalf("expected run count >=1, got %v", runCountRaw)
	}
}

func TestPrometheusIncludesRunLabel(t *testing.T) {
	llm := stubLLM{run: kazi.RunResult{Reply: "ok"}}
	server := newTestServer(t, Options{APIKey: "secret", EnablePrometheus: true}, llm)
	defer server.Close()

	body := map[string]any{"message": "hi"}
	data, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, server.URL+"/run", bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("run request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	req, _ = http.NewRequest(http.MethodGet, server.URL+"/prometheus", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("prometheus request: %v", err)
	}
	bodyBytes, _ := io.ReadAll(resp.Body)
	text := string(bodyBytes)
	if !strings.Contains(text, "kazi_http_requests_total") {
		t.Fatalf("expected prometheus output")
	}
	if !strings.Contains(text, "route=\"run\"") {
		t.Fatalf("expected run route label, got %s", text)
	}
}

func TestRequestIDLogged(t *testing.T) {
	logger := &testLogger{}
	server := newServerWithLogger(t, Options{LogRequests: true, LogSampleRate: 1.0}, nil, logger)
	defer server.Close()

	req, _ := http.NewRequest(http.MethodGet, server.URL+"/health", nil)
	req.Header.Set("X-Request-ID", "abc-123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	_, _ = io.ReadAll(resp.Body)

	logger.mu.Lock()
	defer logger.mu.Unlock()
	if len(logger.infos) == 0 {
		t.Fatalf("expected request logs")
	}
	if logger.infos[0]["request_id"] != "abc-123" {
		t.Fatalf("expected request_id abc-123, got %v", logger.infos[0]["request_id"])
	}
}

type blockingLLM struct {
	ready chan<- struct{}
	block <-chan struct{}
}

func (b blockingLLM) Run(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (kazi.RunResult, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	close(b.ready)
	<-b.block
	return kazi.RunResult{Reply: "ok"}, nil
}

func (b blockingLLM) Stream(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan string, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	ch := make(chan string)
	close(ch)
	return ch, nil
}

func (b blockingLLM) StreamEvents(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan kazi.StreamEvent, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	ch := make(chan kazi.StreamEvent)
	close(ch)
	return ch, nil
}

type silentLLM struct{}

func (s silentLLM) Run(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (kazi.RunResult, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	return kazi.RunResult{Reply: "ok"}, nil
}

func (s silentLLM) Stream(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan string, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	ch := make(chan string)
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	return ch, nil
}

func (s silentLLM) StreamEvents(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan kazi.StreamEvent, error) {
	_ = ctx
	_ = input
	_ = opts
	_ = reg
	ch := make(chan kazi.StreamEvent)
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	return ch, nil
}
