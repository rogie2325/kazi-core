package main

import (
    "flag"
    "fmt"
    "net/http"
    "os"
    "time"

    "github.com/rogie2325/kazi"
    "github.com/rogie2325/kazi/config"
    "github.com/rogie2325/kazi/llm"
    "github.com/rogie2325/kazi/serve"
    "github.com/rogie2325/kazi/tools/builtin"
)

func main() {
    if len(os.Args) < 2 {
        usage()
        os.Exit(1)
    }

    switch os.Args[1] {
    case "serve":
        serveCmd(os.Args[2:])
    case "validate":
        validateCmd(os.Args[2:])
    case "config-schema":
        configSchemaCmd()
    case "-h", "--help", "help":
        usage()
    default:
        fmt.Printf("Unknown command: %q\n", os.Args[1])
        usage()
        os.Exit(1)
    }
}

func usage() {
    fmt.Println("Usage: kazi <command> [args]")
    fmt.Println()
    fmt.Println("Commands:")
    fmt.Println("  serve                  Start the HTTP API server")
    fmt.Println("  validate <kazi.yaml>   Validate config and check provider connectivity")
    fmt.Println("  config-schema          Print the KaziConfig JSON Schema")
}

func serveCmd(args []string) {
    fs := flag.NewFlagSet("serve", flag.ExitOnError)
    configPath := fs.String("config", "", "Path to kazi.yaml")
    addr := fs.String("addr", ":8080", "Listen address")
    apiKey := fs.String("api-key", "", "Bearer token required on all routes")
    prefix := fs.String("prefix", "", "Route prefix, e.g. /api/v1")
    timeout := fs.Duration("timeout", 120*time.Second, "Request timeout")
    maxBody := fs.Int64("max-body-bytes", 1*1024*1024, "Max request body bytes")
    maxConcurrent := fs.Int64("max-concurrent", 50, "Max concurrent requests (0 = unlimited)")
    rateLimit := fs.Int("rate-limit", 0, "Per-IP requests per minute (0 = disabled)")
    enablePrometheus := fs.Bool("prometheus", true, "Expose Prometheus metrics at /prometheus")
    logRequests := fs.Bool("log-requests", true, "Log structured request entries")
    logSampleRate := fs.Float64("log-sample-rate", 1.0, "Sample rate for request logs (0.0-1.0)")
    exampleTools := fs.Bool("example-tools", true, "Register example tools (echo, add)")

    _ = fs.Parse(args)

    cfg := config.DefaultConfig()
    if *configPath != "" {
        loaded, err := config.LoadFile(*configPath)
        if err != nil {
            fmt.Printf("Config parse error: %v\n", err)
            os.Exit(1)
        }
        cfg = loaded
        if err := config.Validate(cfg); err != nil {
            fmt.Printf("Config validation error: %v\n", err)
            os.Exit(1)
        }
    }

    k := kazi.New(cfg)
    client, err := llm.NewFromConfig(cfg.LLM)
    if err != nil {
        fmt.Printf("LLM config error: %v\n", err)
        os.Exit(1)
    }
    k.SetLLMClient(client)
    if *exampleTools {
        if err := builtin.RegisterExampleTools(k.Registry()); err != nil {
            fmt.Printf("Failed to register example tools: %v\n", err)
            os.Exit(1)
        }
    }
    srv := serve.New(k, serve.Options{
        Prefix:         *prefix,
        APIKey:         *apiKey,
        MaxBodyBytes:   *maxBody,
        RequestTimeout: *timeout,
        MaxConcurrent:  *maxConcurrent,
        RateLimitPerMinute: *rateLimit,
        EnablePrometheus: *enablePrometheus,
        LogRequests: *logRequests,
        LogSampleRate: *logSampleRate,
    })

    fmt.Printf("kazi server listening on %s\n", *addr)
    if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
        fmt.Printf("Server error: %v\n", err)
        os.Exit(1)
    }
}

func validateCmd(args []string) {
    if len(args) < 1 {
        fmt.Println("Usage: kazi validate <kazi.yaml>")
        os.Exit(1)
    }

    path := args[0]
    if _, err := os.Stat(path); err != nil {
        fmt.Printf("Config file not found: %s\n", path)
        os.Exit(1)
    }

    cfg, err := config.LoadFile(path)
    if err != nil {
        fmt.Printf("Config parse error: %v\n", err)
        os.Exit(1)
    }

    if err := config.Validate(cfg); err != nil {
        fmt.Printf("Config validation error: %v\n", err)
        os.Exit(1)
    }

    fmt.Println("Config parsed successfully")
    fmt.Printf("  provider : %s\n", cfg.LLM.Provider)
    fmt.Printf("  model    : %s\n", cfg.LLM.Model)

    if key, ok := cfg.LLM.ResolvedAPIKey(); ok {
        fmt.Printf("  api_key  : %s\n", maskKey(key))
    } else {
        fmt.Println("  api_key  : not set")
    }

    fmt.Printf("  rag      : %s / %s\n", cfg.RAG.VectorStore, cfg.RAG.EmbeddingModel)
    fmt.Printf("  memory   : %s\n", cfg.Memory.Backend)

    if len(cfg.MCP.Servers) > 0 {
        fmt.Printf("  mcp      : %d servers\n", len(cfg.MCP.Servers))
    }
    if len(cfg.A2A.DiscoveryEndpoints) > 0 {
        fmt.Printf("  a2a      : %d endpoints\n", len(cfg.A2A.DiscoveryEndpoints))
    }

    fmt.Println()
    fmt.Println("Connectivity checks: not implemented yet")
}

func configSchemaCmd() {
    schema, err := config.SchemaJSON()
    if err != nil {
        fmt.Printf("Schema generation error: %v\n", err)
        os.Exit(1)
    }
    fmt.Println(string(schema))
}

func maskKey(key string) string {
    if len(key) <= 12 {
        return "****"
    }
    return key[:6] + "..." + key[len(key)-4:]
}
