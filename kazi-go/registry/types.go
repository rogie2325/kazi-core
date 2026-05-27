package registry

import "context"

type ToolSource string

const (
    ToolSourceNative ToolSource = "native"
    ToolSourceRAG    ToolSource = "rag"
    ToolSourceMCP    ToolSource = "mcp"
    ToolSourceA2A    ToolSource = "a2a"
)

type ToolParameter struct {
    Name        string   `json:"name"`
    Type        string   `json:"type"`
    Description string   `json:"description"`
    Required    bool     `json:"required"`
    Default     any      `json:"default,omitempty"`
    Enum        []string `json:"enum,omitempty"`
}

type ToolHandler func(ctx context.Context, args map[string]any) (any, error)

type ToolDefinition struct {
    Name        string         `json:"name"`
    Description string         `json:"description"`
    Parameters  []ToolParameter `json:"parameters"`
    Source      ToolSource     `json:"source"`
    Handler     ToolHandler    `json:"-"`
}

type ToolCall struct {
    Name string         `json:"name"`
    Args map[string]any `json:"args"`
}

type ToolResult struct {
    Name   string `json:"name"`
    Result any    `json:"result,omitempty"`
    Error  string `json:"error,omitempty"`
}
