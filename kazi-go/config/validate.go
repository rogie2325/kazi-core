package config

import "strings"

type ValidationError struct {
    Issues []string
}

func (e ValidationError) Error() string {
    return "config validation failed: " + strings.Join(e.Issues, "; ")
}

func Validate(cfg Config) error {
    issues := make([]string, 0)
    if cfg.LLM.Provider == "" {
        issues = append(issues, "llm.provider is required")
    }
    if cfg.LLM.Model == "" {
        issues = append(issues, "llm.model is required")
    }
    if cfg.RAG.ChunkSize <= 0 {
        issues = append(issues, "rag.chunk_size must be > 0")
    }
    if cfg.RAG.ChunkOverlap < 0 {
        issues = append(issues, "rag.chunk_overlap must be >= 0")
    }
    if cfg.Memory.Backend == "" {
        issues = append(issues, "memory.backend is required")
    }
    if len(issues) > 0 {
        return ValidationError{Issues: issues}
    }
    return nil
}
