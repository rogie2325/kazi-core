package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/rogie2325/kazi"
	"github.com/rogie2325/kazi/config"
	"github.com/rogie2325/kazi/registry"
)

const defaultBaseURL = "https://api.openai.com"

var (
	ErrMissingAPIKey = errors.New("openai: api key is required")
)

type Client struct {
	apiKey       string
	baseURL      string
	model        string
	temperature  float64
	maxTokens    int
	seed         *int
	httpClient   *http.Client
	maxRetries   int
	retryBackoff time.Duration
}

func New(cfg config.LLMConfig) (*Client, error) {
	key, ok := cfg.ResolvedAPIKey()
	if !ok || key == "" {
		return nil, ErrMissingAPIKey
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		apiKey:       key,
		baseURL:      strings.TrimSuffix(baseURL, "/"),
		model:        cfg.Model,
		temperature:  cfg.Temperature,
		maxTokens:    cfg.MaxTokens,
		seed:         cfg.Seed,
		httpClient:   &http.Client{Timeout: 60 * time.Second},
		maxRetries:   2,
		retryBackoff: 200 * time.Millisecond,
	}, nil
}

func (c *Client) Run(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (kazi.RunResult, error) {
	if opts.MaxToolCalls > 0 {
		if reg == nil {
			return kazi.RunResult{}, errors.New("openai: tool registry is required")
		}
		return c.runWithTools(ctx, input, opts, reg)
	}

	messages := buildMessages(input, opts.SystemPrompt)
	parsed, err := c.doChat(ctx, messages, nil, opts.UserID)
	if err != nil {
		return kazi.RunResult{}, err
	}
	return buildRunResult(parsed)
}

func (c *Client) Stream(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan string, error) {
	messages := buildMessages(input, opts.SystemPrompt)
	if opts.MaxToolCalls > 0 && reg == nil {
		return nil, errors.New("openai: tool registry is required")
	}

	ch := make(chan string)
	go func() {
		defer close(ch)
		if opts.MaxToolCalls > 0 {
			_ = c.streamTokensWithTools(ctx, messages, opts, reg, func(token string) {
				ch <- token
			})
			return
		}
		_, _, _ = c.streamChat(ctx, messages, nil, opts.UserID, func(token string) {
			ch <- token
		})
	}()
	return ch, nil
}

func (c *Client) StreamEvents(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (<-chan kazi.StreamEvent, error) {
	if opts.MaxToolCalls <= 0 {
		tokenStream, err := c.Stream(ctx, input, opts, reg)
		if err != nil {
			return nil, err
		}

		events := make(chan kazi.StreamEvent)
		go func() {
			defer close(events)
			for token := range tokenStream {
				events <- kazi.StreamEvent{Type: "token", Data: token}
			}
			events <- kazi.StreamEvent{Type: "done", Data: "ok"}
		}()
		return events, nil
	}

	if reg == nil {
		return nil, errors.New("openai: tool registry is required")
	}

	events := make(chan kazi.StreamEvent)
	go func() {
		defer close(events)
		err := c.streamEventsWithTools(ctx, input, opts, reg, events)
		if err != nil {
			events <- kazi.StreamEvent{Type: "error", Data: err.Error()}
		}
	}()
	return events, nil
}

type chatRequest struct {
	Model       string           `json:"model"`
	Messages    []message        `json:"messages"`
	Temperature float64          `json:"temperature,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Seed        *int             `json:"seed,omitempty"`
	User        string           `json:"user,omitempty"`
	Tools       []toolDefinition `json:"tools,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

type message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message message `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type streamResponse struct {
	Choices []struct {
		Delta streamDelta `json:"delta"`
	} `json:"choices"`
}

type streamDelta struct {
	Content   string          `json:"content"`
	ToolCalls []toolCallDelta `json:"tool_calls,omitempty"`
}

type toolCallDelta struct {
	Index    int              `json:"index"`
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

type toolDefinition struct {
	Type     string          `json:"type"`
	Function toolFunctionDef `json:"function"`
}

type toolFunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type toolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type apiError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func buildMessages(input string, systemPrompt string) []message {
	messages := make([]message, 0, 2)
	if systemPrompt != "" {
		messages = append(messages, message{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, message{Role: "user", Content: input})
	return messages
}

func (c *Client) runWithTools(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry) (kazi.RunResult, error) {
	messages := buildMessages(input, opts.SystemPrompt)
	tools := buildTools(reg)
	if len(tools) == 0 {
		return c.runOnce(ctx, messages, opts.UserID)
	}

	for i := 0; i < opts.MaxToolCalls; i++ {
		resp, err := c.doChat(ctx, messages, tools, opts.UserID)
		if err != nil {
			return kazi.RunResult{}, err
		}
		if len(resp.Choices) == 0 {
			return kazi.RunResult{}, errors.New("openai: empty response")
		}
		msg := resp.Choices[0].Message
		if len(msg.ToolCalls) == 0 {
			return buildRunResult(resp)
		}

		messages = append(messages, message{Role: "assistant", Content: msg.Content, ToolCalls: msg.ToolCalls})
		for _, call := range msg.ToolCalls {
			output, _ := executeTool(ctx, reg, call)
			messages = append(messages, message{Role: "tool", ToolCallID: call.ID, Content: output})
		}
	}

	return kazi.RunResult{}, errors.New("openai: tool call limit exceeded")
}

func (c *Client) streamEventsWithTools(ctx context.Context, input string, opts kazi.RunOptions, reg *registry.Registry, events chan<- kazi.StreamEvent) error {
	messages := buildMessages(input, opts.SystemPrompt)
	tools := buildTools(reg)
	if len(tools) == 0 {
		return errors.New("openai: no tools registered")
	}

	for i := 0; i < opts.MaxToolCalls; i++ {
		assistantContent, toolCalls, err := c.streamChat(ctx, messages, tools, opts.UserID, func(token string) {
			events <- kazi.StreamEvent{Type: "token", Data: token}
		})
		if err != nil {
			return err
		}
		if len(toolCalls) == 0 {
			events <- kazi.StreamEvent{Type: "done", Data: "ok"}
			return nil
		}

		messages = append(messages, message{Role: "assistant", Content: assistantContent, ToolCalls: toolCalls})
		for _, call := range toolCalls {
			events <- kazi.StreamEvent{Type: "tool_start", Data: map[string]any{"name": call.Function.Name, "id": call.ID}}
			output, err := executeTool(ctx, reg, call)
			payload := map[string]any{"name": call.Function.Name, "id": call.ID, "output": output}
			if err != nil {
				payload["error"] = err.Error()
			}
			events <- kazi.StreamEvent{Type: "tool_end", Data: payload}
			messages = append(messages, message{Role: "tool", ToolCallID: call.ID, Content: output})
		}
	}

	return errors.New("openai: tool call limit exceeded")
}

func (c *Client) streamTokensWithTools(ctx context.Context, messages []message, opts kazi.RunOptions, reg *registry.Registry, onToken func(string)) error {
	tools := buildTools(reg)
	if len(tools) == 0 {
		return errors.New("openai: no tools registered")
	}

	for i := 0; i < opts.MaxToolCalls; i++ {
		assistantContent, toolCalls, err := c.streamChat(ctx, messages, tools, opts.UserID, onToken)
		if err != nil {
			return err
		}
		if len(toolCalls) == 0 {
			return nil
		}

		messages = append(messages, message{Role: "assistant", Content: assistantContent, ToolCalls: toolCalls})
		for _, call := range toolCalls {
			output, _ := executeTool(ctx, reg, call)
			messages = append(messages, message{Role: "tool", ToolCallID: call.ID, Content: output})
		}
	}

	return errors.New("openai: tool call limit exceeded")
}

func (c *Client) streamChat(ctx context.Context, messages []message, tools []toolDefinition, userID string, onToken func(string)) (string, []toolCall, error) {
	req := chatRequest{
		Model:       c.model,
		Temperature: c.temperature,
		MaxTokens:   c.maxTokens,
		Seed:        c.seed,
		Messages:    messages,
		Tools:       tools,
		User:        userID,
		Stream:      true,
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return "", nil, err
	}

	resp, err := c.doRequest(ctx, payload, true)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	var contentBuilder strings.Builder
	toolCalls := map[int]*toolCall{}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk streamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			contentBuilder.WriteString(delta.Content)
			if onToken != nil {
				onToken(delta.Content)
			}
		}
		for _, toolDelta := range delta.ToolCalls {
			entry, ok := toolCalls[toolDelta.Index]
			if !ok {
				entry = &toolCall{Type: toolDelta.Type}
				toolCalls[toolDelta.Index] = entry
			}
			if toolDelta.Type != "" {
				entry.Type = toolDelta.Type
			}
			if toolDelta.ID != "" {
				entry.ID = toolDelta.ID
			}
			if toolDelta.Function.Name != "" {
				entry.Function.Name = toolDelta.Function.Name
			}
			if toolDelta.Function.Arguments != "" {
				entry.Function.Arguments += toolDelta.Function.Arguments
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", nil, err
	}

	ordered := make([]toolCall, 0, len(toolCalls))
	for i := 0; i < len(toolCalls); i++ {
		if call, ok := toolCalls[i]; ok {
			ordered = append(ordered, *call)
		}
	}
	return contentBuilder.String(), ordered, nil
}

func (c *Client) runOnce(ctx context.Context, messages []message, userID string) (kazi.RunResult, error) {
	parsed, err := c.doChat(ctx, messages, nil, userID)
	if err != nil {
		return kazi.RunResult{}, err
	}
	return buildRunResult(parsed)
}

func (c *Client) doChat(ctx context.Context, messages []message, tools []toolDefinition, userID string) (chatResponse, error) {
	req := chatRequest{
		Model:       c.model,
		Temperature: c.temperature,
		MaxTokens:   c.maxTokens,
		Seed:        c.seed,
		Messages:    messages,
		Tools:       tools,
		User:        userID,
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return chatResponse{}, err
	}

	resp, err := c.doRequest(ctx, payload, false)
	if err != nil {
		return chatResponse{}, err
	}
	defer resp.Body.Close()

	var parsed chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return chatResponse{}, err
	}
	return parsed, nil
}

func buildRunResult(parsed chatResponse) (kazi.RunResult, error) {
	if len(parsed.Choices) == 0 {
		return kazi.RunResult{}, errors.New("openai: empty response")
	}
	reply := parsed.Choices[0].Message.Content
	result := kazi.RunResult{Reply: reply}
	if parsed.Usage.PromptTokens > 0 {
		result.InputTokens = &parsed.Usage.PromptTokens
	}
	if parsed.Usage.CompletionTokens > 0 {
		result.OutputTokens = &parsed.Usage.CompletionTokens
	}
	return result, nil
}

func buildTools(reg *registry.Registry) []toolDefinition {
	if reg == nil {
		return nil
	}
	tools := reg.List()
	if len(tools) == 0 {
		return nil
	}
	out := make([]toolDefinition, 0, len(tools))
	for _, tool := range tools {
		out = append(out, toolDefinition{
			Type: "function",
			Function: toolFunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  buildToolParameters(tool.Parameters),
			},
		})
	}
	return out
}

func buildToolParameters(params []registry.ToolParameter) map[string]any {
	properties := map[string]any{}
	required := make([]string, 0)
	for _, param := range params {
		schema := map[string]any{
			"type": normalizeType(param.Type),
		}
		if param.Description != "" {
			schema["description"] = param.Description
		}
		if len(param.Enum) > 0 {
			schema["enum"] = param.Enum
		}
		if param.Default != nil {
			schema["default"] = param.Default
		}
		properties[param.Name] = schema
		if param.Required {
			required = append(required, param.Name)
		}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func normalizeType(value string) string {
	switch strings.ToLower(value) {
	case "string", "number", "integer", "boolean", "object", "array":
		return strings.ToLower(value)
	case "bool":
		return "boolean"
	case "int":
		return "integer"
	case "float", "float64", "double":
		return "number"
	default:
		return "string"
	}
}

func executeTool(ctx context.Context, reg *registry.Registry, call toolCall) (string, error) {
	if reg == nil {
		return marshalToolOutput(nil, errors.New("tool registry not configured")), nil
	}
	args := map[string]any{}
	if call.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return marshalToolOutput(nil, fmt.Errorf("invalid tool args: %w", err)), nil
		}
	}
	result, err := reg.Call(ctx, call.Function.Name, args)
	return marshalToolOutput(result, err), nil
}

func marshalToolOutput(result any, err error) string {
	payload := map[string]any{}
	if err != nil {
		payload["error"] = err.Error()
	} else {
		payload["result"] = result
	}
	data, marshalErr := json.Marshal(payload)
	if marshalErr != nil {
		return fmt.Sprintf("{\"error\":%q}", marshalErr.Error())
	}
	return string(data)
}

func parseAPIError(body io.Reader, status int) error {
	data, _ := io.ReadAll(body)
	if len(data) == 0 {
		return fmt.Errorf("openai: request failed with status %d", status)
	}

	var parsed apiError
	if err := json.Unmarshal(data, &parsed); err == nil && parsed.Error.Message != "" {
		return fmt.Errorf("openai: %s", parsed.Error.Message)
	}

	return fmt.Errorf("openai: request failed with status %d", status)
}

func (c *Client) doRequest(ctx context.Context, payload []byte, stream bool) (*http.Response, error) {
	endpoint := c.baseURL + "/v1/chat/completions"
	attempt := 0
	for {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		if stream {
			httpReq.Header.Set("Accept", "text/event-stream")
		}

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			if !c.shouldRetry(err, 0) || attempt >= c.maxRetries {
				return nil, err
			}
			if err := c.sleepBackoff(ctx, attempt); err != nil {
				return nil, err
			}
			attempt++
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp, nil
		}

		if !isRetryableStatus(resp.StatusCode) || attempt >= c.maxRetries {
			err = parseAPIError(resp.Body, resp.StatusCode)
			return nil, err
		}

		drainBody(resp.Body)
		if err := c.sleepBackoff(ctx, attempt); err != nil {
			return nil, err
		}
		attempt++
	}
}

func (c *Client) shouldRetry(err error, status int) bool {
	if err != nil {
		return true
	}
	return isRetryableStatus(status)
}

func isRetryableStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusRequestTimeout,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func (c *Client) sleepBackoff(ctx context.Context, attempt int) error {
	delay := c.retryBackoff * time.Duration(1<<attempt)
	if delay > 2*time.Second {
		delay = 2 * time.Second
	}
	jitter := time.Duration(rand.Int63n(int64(delay / 2)))
	delay += jitter
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func drainBody(body io.ReadCloser) {
	_, _ = io.Copy(io.Discard, body)
	_ = body.Close()
}
