package builtin

import (
    "context"
    "encoding/json"
    "errors"

    "github.com/rogie2325/kazi/registry"
)

func RegisterExampleTools(reg *registry.Registry) error {
    if reg == nil {
        return errors.New("registry is required")
    }
    if err := reg.Register(exampleEchoTool()); err != nil {
        return err
    }
    if err := reg.Register(exampleAddTool()); err != nil {
        return err
    }
    return nil
}

func exampleEchoTool() registry.ToolDefinition {
    return registry.ToolDefinition{
        Name:        "echo",
        Description: "Echo a message for debugging tool calls.",
        Source:      registry.ToolSourceNative,
        Parameters: []registry.ToolParameter{
            {
                Name:        "message",
                Type:        "string",
                Description: "Message to echo back.",
                Required:    true,
            },
        },
        Handler: func(ctx context.Context, args map[string]any) (any, error) {
            _ = ctx
            raw, ok := args["message"]
            if !ok {
                return nil, errors.New("message is required")
            }
            msg, ok := raw.(string)
            if !ok {
                return nil, errors.New("message must be a string")
            }
            return map[string]any{"message": msg}, nil
        },
    }
}

func exampleAddTool() registry.ToolDefinition {
    return registry.ToolDefinition{
        Name:        "add",
        Description: "Add two numbers.",
        Source:      registry.ToolSourceNative,
        Parameters: []registry.ToolParameter{
            {
                Name:        "a",
                Type:        "number",
                Description: "First operand.",
                Required:    true,
            },
            {
                Name:        "b",
                Type:        "number",
                Description: "Second operand.",
                Required:    true,
            },
        },
        Handler: func(ctx context.Context, args map[string]any) (any, error) {
            _ = ctx
            a, ok := parseNumber(args, "a")
            if !ok {
                return nil, errors.New("a must be a number")
            }
            b, ok := parseNumber(args, "b")
            if !ok {
                return nil, errors.New("b must be a number")
            }
            return map[string]any{"result": a + b}, nil
        },
    }
}

func parseNumber(args map[string]any, key string) (float64, bool) {
    raw, ok := args[key]
    if !ok {
        return 0, false
    }
    switch v := raw.(type) {
    case float64:
        return v, true
    case int:
        return float64(v), true
    case int32:
        return float64(v), true
    case int64:
        return float64(v), true
    case json.Number:
        parsed, err := v.Float64()
        if err != nil {
            return 0, false
        }
        return parsed, true
    default:
        return 0, false
    }
}
