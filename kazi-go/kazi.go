// this is my very first go package, so please be gentle with me :)

package kazi

import (
    "context"
    "errors"

    "github.com/rogie2325/kazi/config"
    "github.com/rogie2325/kazi/registry"
)

type Kazi struct {
    cfg config.Config
    registry *registry.Registry
    llm LLMClient
}

var (
    ErrNotImplemented = errors.New("kazi: not implemented")
    ErrNoLLM          = errors.New("kazi: llm not configured")
)

func New(cfg config.Config) *Kazi {
    return &Kazi{cfg: cfg, registry: registry.New()}
}

func NewDefault() *Kazi {
    return New(config.DefaultConfig())
}

func (k *Kazi) Config() config.Config {
    return k.cfg
}

func (k *Kazi) Registry() *registry.Registry {
    return k.registry
}

func (k *Kazi) RegisterTool(tool registry.ToolDefinition) error {
    return k.registry.Register(tool)
}

func (k *Kazi) SetLLMClient(client LLMClient) {
    k.llm = client
}

func (k *Kazi) HasLLM() bool {
    return k.llm != nil
}

type RunOptions struct {
    ThreadID string
    UserToken string
    TenantID string
    UserID string
    SystemPrompt string
    MaxToolCalls int
    TrackCost bool
}

type RunOption func(*RunOptions)

func WithThreadID(id string) RunOption {
    return func(o *RunOptions) {
        o.ThreadID = id
    }
}

func WithUserToken(token string) RunOption {
    return func(o *RunOptions) {
        o.UserToken = token
    }
}

func WithTenantID(id string) RunOption {
    return func(o *RunOptions) {
        o.TenantID = id
    }
}

func WithUserID(id string) RunOption {
    return func(o *RunOptions) {
        o.UserID = id
    }
}

func WithSystemPrompt(prompt string) RunOption {
    return func(o *RunOptions) {
        o.SystemPrompt = prompt
    }
}

func WithMaxToolCalls(limit int) RunOption {
    return func(o *RunOptions) {
        o.MaxToolCalls = limit
    }
}

func WithTrackCost(track bool) RunOption {
    return func(o *RunOptions) {
        o.TrackCost = track
    }
}

type RunResult struct {
    Reply string
    CostUSD *float64
    InputTokens *int
    OutputTokens *int
}

type StreamEvent struct {
    Type string `json:"type"`
    Data any    `json:"data"`
}

type HealthCheck struct {
    Status string
    Error string
    LatencyMs int
}

type HealthReport struct {
    Status string
    Checks map[string]HealthCheck
}

type LLMClient interface {
    Run(ctx context.Context, input string, opts RunOptions, reg *registry.Registry) (RunResult, error)
    Stream(ctx context.Context, input string, opts RunOptions, reg *registry.Registry) (<-chan string, error)
    StreamEvents(ctx context.Context, input string, opts RunOptions, reg *registry.Registry) (<-chan StreamEvent, error)
}

func (k *Kazi) Run(ctx context.Context, input string, opts ...RunOption) (RunResult, error) {
    if k.llm == nil {
        return RunResult{}, ErrNoLLM
    }
    options := applyRunOptions(opts...)
    return k.llm.Run(ctx, input, options, k.registry)
}

func (k *Kazi) Stream(ctx context.Context, input string, opts ...RunOption) (<-chan string, error) {
    if k.llm == nil {
        return nil, ErrNoLLM
    }
    options := applyRunOptions(opts...)
    return k.llm.Stream(ctx, input, options, k.registry)
}

func (k *Kazi) StreamEvents(ctx context.Context, input string, opts ...RunOption) (<-chan StreamEvent, error) {
    if k.llm == nil {
        return nil, ErrNoLLM
    }
    options := applyRunOptions(opts...)
    return k.llm.StreamEvents(ctx, input, options, k.registry)
}

func (k *Kazi) Ingest(ctx context.Context, path string, indexName string) error {
    _ = ctx
    _ = path
    _ = indexName
    return ErrNotImplemented
}

func (k *Kazi) Health(ctx context.Context) (HealthReport, error) {
    _ = ctx
    return HealthReport{Status: "healthy"}, nil
}

func applyRunOptions(opts ...RunOption) RunOptions {
    options := RunOptions{ThreadID: "default"}
    for _, opt := range opts {
        if opt != nil {
            opt(&options)
        }
    }
    if options.ThreadID == "" {
        options.ThreadID = "default"
    }
    return options
}
