package registry

import (
	"encoding/json"
	"testing"
)

func TestValidateArgsDefaultsEnumsAndTypes(t *testing.T) {
	tool := ToolDefinition{
		Name: "example",
		Parameters: []ToolParameter{
			{Name: "mode", Type: "string", Required: true, Enum: []string{"fast", "slow"}},
			{Name: "count", Type: "integer", Default: 2},
		},
	}

	args := map[string]any{"mode": "fast"}
	if err := ValidateArgs(tool, args); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if args["count"] != 2 {
		t.Fatalf("expected default count=2, got %v", args["count"])
	}

	badEnum := map[string]any{"mode": "medium"}
	if err := ValidateArgs(tool, badEnum); err == nil {
		t.Fatalf("expected enum error")
	}

	badType := map[string]any{"mode": "fast", "count": 1.25}
	if err := ValidateArgs(tool, badType); err == nil {
		t.Fatalf("expected type error")
	}

	jsonArgs := map[string]any{"mode": "slow", "count": json.Number("3")}
	if err := ValidateArgs(tool, jsonArgs); err != nil {
		t.Fatalf("unexpected json number error: %v", err)
	}
}
