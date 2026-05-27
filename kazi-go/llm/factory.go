package llm

import (
    "fmt"

    "github.com/rogie2325/kazi"
    "github.com/rogie2325/kazi/config"
    "github.com/rogie2325/kazi/llm/openai"
)

func NewFromConfig(cfg config.LLMConfig) (kazi.LLMClient, error) {
    switch cfg.Provider {
    case config.LLMProviderOpenAI:
        return openai.New(cfg)
    case config.LLMProviderAnthropic, config.LLMProviderGoogle, config.LLMProviderLocal:
        return nil, fmt.Errorf("llm provider not implemented: %s", cfg.Provider)
    default:
        return nil, fmt.Errorf("unknown llm provider: %s", cfg.Provider)
    }
}
