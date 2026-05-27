package serve

type RunRequest struct {
    Message      string `json:"message"`
    ThreadID     string `json:"thread_id,omitempty"`
    SystemPrompt string `json:"system_prompt,omitempty"`
    MaxToolCalls int    `json:"max_tool_calls,omitempty"`
    TrackCost    bool   `json:"track_cost,omitempty"`
    UserID       string `json:"user_id,omitempty"`
    TenantID     string `json:"tenant_id,omitempty"`
}

type RunResponse struct {
    Reply        string   `json:"reply"`
    CostUSD      *float64 `json:"cost_usd,omitempty"`
    InputTokens  *int     `json:"input_tokens,omitempty"`
    OutputTokens *int     `json:"output_tokens,omitempty"`
}

type IngestRequest struct {
    Path      string `json:"path"`
    IndexName string `json:"index_name,omitempty"`
}

type ErrorResponse struct {
    Error string `json:"error"`
}
