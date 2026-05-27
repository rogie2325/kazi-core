package registry

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "math"
    "strings"
    "sync"
    "sync/atomic"
)

var (
    ErrToolNotFound      = errors.New("tool not found")
    ErrToolHandlerMissing = errors.New("tool handler missing")
)

type ArgError struct {
    Message string
}

func (e ArgError) Error() string {
    return e.Message
}

type Registry struct {
    mu    sync.RWMutex
    tools map[string]ToolDefinition
    calls atomic.Int64
    errors atomic.Int64
}

func New() *Registry {
    return &Registry{tools: map[string]ToolDefinition{}}
}

func (r *Registry) Register(tool ToolDefinition) error {
    if tool.Name == "" {
        return ArgError{Message: "tool name is required"}
    }
    r.mu.Lock()
    defer r.mu.Unlock()
    r.tools[tool.Name] = tool
    return nil
}

func (r *Registry) Get(name string) (ToolDefinition, bool) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    tool, ok := r.tools[name]
    return tool, ok
}

func (r *Registry) List() []ToolDefinition {
    r.mu.RLock()
    defer r.mu.RUnlock()
    out := make([]ToolDefinition, 0, len(r.tools))
    for _, tool := range r.tools {
        out = append(out, tool)
    }
    return out
}

func (r *Registry) Call(ctx context.Context, name string, args map[string]any) (any, error) {
    tool, ok := r.Get(name)
    if !ok {
        r.calls.Add(1)
        r.errors.Add(1)
        return nil, ErrToolNotFound
    }
    if tool.Handler == nil {
        r.calls.Add(1)
        r.errors.Add(1)
        return nil, ErrToolHandlerMissing
    }
    if err := ValidateArgs(tool, args); err != nil {
        r.calls.Add(1)
        r.errors.Add(1)
        return nil, err
    }
    r.calls.Add(1)
    result, err := tool.Handler(ctx, args)
    if err != nil {
        r.errors.Add(1)
    }
    return result, err
}

type RegistryStats struct {
    Calls  int64
    Errors int64
}

func (r *Registry) Stats() RegistryStats {
    return RegistryStats{
        Calls:  r.calls.Load(),
        Errors: r.errors.Load(),
    }
}

func ValidateArgs(tool ToolDefinition, args map[string]any) error {
    if args == nil {
        args = map[string]any{}
    }
    required := make(map[string]ToolParameter)
    params := make(map[string]ToolParameter)
    for _, param := range tool.Parameters {
        params[param.Name] = param
        if param.Required {
            required[param.Name] = param
        }
    }
    for name, param := range params {
        value, ok := args[name]
        if !ok {
            if param.Default != nil {
                args[name] = param.Default
                continue
            }
            if param.Required {
                return ArgError{Message: fmt.Sprintf("tool '%s' missing required arg: %s", tool.Name, name)}
            }
            continue
        }
        if len(param.Enum) > 0 {
            strValue, ok := value.(string)
            if !ok {
                return ArgError{Message: fmt.Sprintf("tool '%s' arg '%s' must be a string", tool.Name, name)}
            }
            if !containsString(param.Enum, strValue) {
                return ArgError{Message: fmt.Sprintf("tool '%s' arg '%s' must be one of %v", tool.Name, name, param.Enum)}
            }
        }
        if param.Type != "" && !validateType(value, param.Type) {
            return ArgError{Message: fmt.Sprintf("tool '%s' arg '%s' must be %s", tool.Name, name, param.Type)}
        }
    }
    return nil
}

func containsString(values []string, target string) bool {
    for _, value := range values {
        if value == target {
            return true
        }
    }
    return false
}

func validateType(value any, expected string) bool {
    switch normalizeType(expected) {
    case "string":
        _, ok := value.(string)
        return ok
    case "boolean":
        _, ok := value.(bool)
        return ok
    case "number":
        return isNumber(value)
    case "integer":
        return isInteger(value)
    case "object":
        _, ok := value.(map[string]any)
        return ok
    case "array":
        _, ok := value.([]any)
        return ok
    default:
        return true
    }
}

func isNumber(value any) bool {
    switch v := value.(type) {
    case float64, float32, int, int32, int64, uint, uint32, uint64:
        return true
    case json.Number:
        _, err := v.Float64()
        return err == nil
    default:
        return false
    }
}

func isInteger(value any) bool {
    switch v := value.(type) {
    case int, int32, int64, uint, uint32, uint64:
        return true
    case float64:
        return math.Mod(v, 1) == 0
    case float32:
        return math.Mod(float64(v), 1) == 0
    case json.Number:
        parsed, err := v.Float64()
        if err != nil {
            return false
        }
        return math.Mod(parsed, 1) == 0
    default:
        return false
    }
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
