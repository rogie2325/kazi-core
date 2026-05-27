package serve

import (
    "context"
    "crypto/rand"
    "crypto/subtle"
    "encoding/binary"
    "encoding/hex"
    "encoding/json"
    "errors"
    "fmt"
    "net"
    "net/http"
    "os"
    "strings"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rogie2325/kazi"
)

type Options struct {
    Prefix         string
    APIKey         string
    MaxBodyBytes   int64
    RequestTimeout time.Duration
    MaxConcurrent  int64
    RateLimitPerMinute int
    Logger         Logger
    LogRequests    bool
    EnablePrometheus bool
    LogSampleRate  float64
}

func DefaultOptions() Options {
    return Options{
        Prefix:       "",
        MaxBodyBytes: 1 * 1024 * 1024,
        RequestTimeout: 120 * time.Second,
        MaxConcurrent: 50,
        LogRequests: true,
        EnablePrometheus: true,
        LogSampleRate: 1.0,
    }
}

type routeMetrics struct {
    count      atomic.Int64
    errors     atomic.Int64
    latencyMs  atomic.Int64
}

type rateEntry struct {
    window time.Time
    count  int
}

type Server struct {
    kazi *kazi.Kazi
    opts Options
    mux  *http.ServeMux

    activeRequests atomic.Int64
    metrics map[string]*routeMetrics
    logger Logger
    prom   *promMetrics

    rateMu sync.Mutex
    rate   map[string]*rateEntry
}

func New(k *kazi.Kazi, opts Options) *Server {
    if opts.MaxBodyBytes <= 0 {
        opts.MaxBodyBytes = 1 * 1024 * 1024
    }
    if opts.RequestTimeout < 0 {
        opts.RequestTimeout = 0
    }
    if opts.MaxConcurrent < 0 {
        opts.MaxConcurrent = 0
    }
    if opts.LogSampleRate < 0 {
        opts.LogSampleRate = 0
    }
    if opts.LogSampleRate > 1 {
        opts.LogSampleRate = 1
    }
    if opts.Logger == nil {
        opts.Logger = NewJSONLogger(os.Stdout)
    }
    if opts.Prefix != "" {
        if !strings.HasPrefix(opts.Prefix, "/") {
            opts.Prefix = "/" + opts.Prefix
        }
        opts.Prefix = strings.TrimSuffix(opts.Prefix, "/")
    }

    s := &Server{
        kazi: k,
        opts: opts,
        mux:  http.NewServeMux(),
        metrics: map[string]*routeMetrics{},
        rate:   map[string]*rateEntry{},
        logger: opts.Logger,
    }
    if opts.EnablePrometheus {
        s.prom = newPromMetrics()
    }
    s.metrics["run"] = &routeMetrics{}
    s.metrics["stream"] = &routeMetrics{}
    s.metrics["events"] = &routeMetrics{}
    s.metrics["ingest"] = &routeMetrics{}
    s.metrics["health"] = &routeMetrics{}
    s.metrics["metrics"] = &routeMetrics{}
    s.metrics["prometheus"] = &routeMetrics{}
    s.routes()
    return s
}

func (s *Server) Handler() http.Handler {
    return s.mux
}

func (s *Server) routes() {
    prefix := s.opts.Prefix
    s.handle("POST", prefix+"/run", "run", s.handleRun)
    s.handle("POST", prefix+"/stream", "stream", s.handleStream)
    s.handle("POST", prefix+"/events", "events", s.handleEvents)
    s.handle("POST", prefix+"/ingest", "ingest", s.handleIngest)
    s.handle("GET", prefix+"/health", "health", s.handleHealth)
    s.handle("GET", prefix+"/metrics", "metrics", s.handleMetrics)
    if s.prom != nil {
        s.handle("GET", prefix+"/prometheus", "prometheus", s.handlePrometheus)
    }
}

func (s *Server) handle(method, path, name string, handler http.HandlerFunc) {
    s.mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
        sw := newStatusWriter(w)
        s.applySecurityHeaders(sw)
        requestID := s.ensureRequestID(sw, r)
        start := time.Now()
        ip := s.clientIP(r)
        r = r.WithContext(withRequestID(r.Context(), requestID))
        defer func() {
            duration := time.Since(start)
            s.recordLatency(name, duration)
            s.observeProm(name, r.Method, sw.Status(), duration)
            if s.opts.LogRequests && s.logger != nil && s.shouldSample() {
                s.logRequest(name, r, sw, requestID, ip, duration)
            }
        }()
        if r.Method != method {
            s.recordError(name)
            sw.WriteHeader(http.StatusMethodNotAllowed)
            return
        }
        if s.opts.MaxConcurrent > 0 && s.activeRequests.Load() >= s.opts.MaxConcurrent {
            s.recordError(name)
            s.writeJSON(sw, http.StatusServiceUnavailable, ErrorResponse{Error: "server busy"})
            return
        }
        if !s.checkRateLimit(ip) {
            s.recordError(name)
            s.writeJSON(sw, http.StatusTooManyRequests, ErrorResponse{Error: "rate limit exceeded"})
            return
        }
        s.recordRequest(name)
        s.activeRequests.Add(1)
        s.setActiveGauge()
        defer func() {
            s.activeRequests.Add(-1)
            s.setActiveGauge()
        }()
        handler(sw, r)
    })
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
    if !s.isAuthed(r) {
        s.recordError("run")
        s.writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "unauthorized"})
        return
    }

    var req RunRequest
    if err := s.readJSON(w, r, &req); err != nil {
        s.recordError("run")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: err.Error()})
        return
    }
    sanitizeRunRequest(&req)
    if req.Message == "" {
        s.recordError("run")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "message is required"})
        return
    }

    ctx, cancel := s.withTimeout(r.Context())
    defer cancel()
    result, err := s.kazi.Run(ctx, req.Message, buildRunOptions(req)...)
    if err != nil {
        s.recordError("run")
        s.writeRunError(w, err)
        return
    }

    resp := RunResponse{
        Reply:        result.Reply,
        CostUSD:      result.CostUSD,
        InputTokens:  result.InputTokens,
        OutputTokens: result.OutputTokens,
    }
    s.writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
    if !s.isAuthed(r) {
        s.recordError("stream")
        s.writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "unauthorized"})
        return
    }

    var req RunRequest
    if err := s.readJSON(w, r, &req); err != nil {
        s.recordError("stream")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: err.Error()})
        return
    }
    sanitizeRunRequest(&req)
    if req.Message == "" {
        s.recordError("stream")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "message is required"})
        return
    }

    ctx, cancel := s.withTimeout(r.Context())
    defer cancel()
    ch, err := s.kazi.StreamEvents(ctx, req.Message, buildRunOptions(req)...)
    if err != nil {
        s.recordError("stream")
        s.writeRunError(w, err)
        return
    }

    if err := s.streamTokensFromEvents(w, ctx, ch); err != nil {
        s.recordError("stream")
        return
    }
}

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
    if !s.isAuthed(r) {
        s.recordError("events")
        s.writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "unauthorized"})
        return
    }

    var req RunRequest
    if err := s.readJSON(w, r, &req); err != nil {
        s.recordError("events")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: err.Error()})
        return
    }
    sanitizeRunRequest(&req)
    if req.Message == "" {
        s.recordError("events")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "message is required"})
        return
    }

    ctx, cancel := s.withTimeout(r.Context())
    defer cancel()
    ch, err := s.kazi.StreamEvents(ctx, req.Message, buildRunOptions(req)...)
    if err != nil {
        s.recordError("events")
        s.writeRunError(w, err)
        return
    }

    if err := s.streamEvents(w, ctx, ch); err != nil {
        s.recordError("events")
        return
    }
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
    if !s.isAuthed(r) {
        s.recordError("ingest")
        s.writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "unauthorized"})
        return
    }

    var req IngestRequest
    if err := s.readJSON(w, r, &req); err != nil {
        s.recordError("ingest")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: err.Error()})
        return
    }
    req.Path = sanitizeString(req.Path, 1024)
    req.IndexName = sanitizeString(req.IndexName, 128)
    if req.Path == "" {
        s.recordError("ingest")
        s.writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "path is required"})
        return
    }
    if req.IndexName == "" {
        req.IndexName = "default"
    }

    ctx, cancel := s.withTimeout(r.Context())
    defer cancel()
    if err := s.kazi.Ingest(ctx, req.Path, req.IndexName); err != nil {
        s.recordError("ingest")
        s.writeRunError(w, err)
        return
    }

    s.writeJSON(w, http.StatusOK, map[string]string{
        "status": "ok",
        "path":   req.Path,
        "index":  req.IndexName,
    })
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := s.withTimeout(r.Context())
    defer cancel()
    report, err := s.kazi.Health(ctx)
    if err != nil {
        s.recordError("health")
        s.writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "health check failed"})
        return
    }

    statusCode := http.StatusOK
    if report.Status != "healthy" && report.Status != "degraded" {
        statusCode = http.StatusServiceUnavailable
    }

    if s.isAuthed(r) || s.opts.APIKey == "" {
        s.writeJSON(w, statusCode, report)
        return
    }

    s.writeJSON(w, statusCode, map[string]string{"status": report.Status})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
    if !s.isAuthed(r) {
        s.recordError("metrics")
        s.writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "unauthorized"})
        return
    }

    cfg := s.kazi.Config()
    tools := s.kazi.Registry().List()
    toolSources := map[string]int{}
    for _, tool := range tools {
        toolSources[string(tool.Source)] += 1
    }
    stats := s.kazi.Registry().Stats()
    if s.prom != nil {
        s.prom.SetToolStats(len(tools), stats.Calls, stats.Errors)
    }
    requestCounts := map[string]int64{}
    requestErrors := map[string]int64{}
    latencyMs := map[string]float64{}
    for name, metric := range s.metrics {
        if metric == nil {
            continue
        }
        count := metric.count.Load()
        totalLatency := metric.latencyMs.Load()
        requestCounts[name] = count
        requestErrors[name] = metric.errors.Load()
        if count > 0 {
            latencyMs[name] = float64(totalLatency) / float64(count)
        }
    }
    payload := map[string]any{
        "llm_configured":           s.kazi.HasLLM(),
        "llm_provider":             cfg.LLM.Provider,
        "llm_model":                cfg.LLM.Model,
        "tools_registered":         len(tools),
        "tool_sources":             toolSources,
        "tool_calls_total":         stats.Calls,
        "tool_errors_total":        stats.Errors,
        "voice_enabled":             cfg.Voice != nil,
        "memory_backend":            cfg.Memory.Backend,
        "semantic_cache_enabled":    cfg.SemanticCache != nil,
        "guardrails_enabled":        cfg.Guardrails != nil,
        "tool_result_cache_ttl":     cfg.ToolResultCacheTTLSeconds,
        "request_timeout_seconds":   int(s.opts.RequestTimeout.Seconds()),
        "max_body_bytes":            s.opts.MaxBodyBytes,
        "max_concurrent":            s.opts.MaxConcurrent,
        "rate_limit_per_minute":     s.opts.RateLimitPerMinute,
        "active_requests":           s.activeRequests.Load(),
        "requests_total":            requestCounts,
        "request_errors":            requestErrors,
        "request_avg_latency_ms":    latencyMs,
        "mcp_servers":               len(cfg.MCP.Servers),
        "a2a_endpoints":             len(cfg.A2A.DiscoveryEndpoints),
    }

    s.writeJSON(w, http.StatusOK, payload)
}

func (s *Server) handlePrometheus(w http.ResponseWriter, r *http.Request) {
    if s.prom == nil {
        s.writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "prometheus disabled"})
        return
    }
    if !s.isAuthed(r) {
        s.recordError("prometheus")
        s.writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "unauthorized"})
        return
    }
    s.prom.Handler().ServeHTTP(w, r)
}

func (s *Server) streamTokensFromEvents(w http.ResponseWriter, ctx context.Context, ch <-chan kazi.StreamEvent) error {
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("X-Accel-Buffering", "no")

    flusher, ok := w.(http.Flusher)
    if !ok {
        s.writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "streaming not supported"})
        return errors.New("streaming not supported")
    }

    for {
        select {
        case <-ctx.Done():
            _ = writeSSE(w, map[string]string{"error": "timeout"})
            flusher.Flush()
            return ctx.Err()
        case event, ok := <-ch:
            if !ok {
                _, _ = fmt.Fprint(w, "data: [DONE]\n\n")
                flusher.Flush()
                return nil
            }
            switch event.Type {
            case "token":
                if err := writeSSE(w, map[string]any{"token": event.Data}); err != nil {
                    return err
                }
            case "error":
                s.logStreamError(ctx, "stream", event.Data)
                _ = writeSSE(w, map[string]any{"error": event.Data})
                flusher.Flush()
                return errors.New("stream error")
            case "done":
                _, _ = fmt.Fprint(w, "data: [DONE]\n\n")
                flusher.Flush()
                return nil
            default:
                // Ignore tool events in /stream.
            }
            flusher.Flush()
        }
    }
}

func (s *Server) streamEvents(w http.ResponseWriter, ctx context.Context, ch <-chan kazi.StreamEvent) error {
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("X-Accel-Buffering", "no")

    flusher, ok := w.(http.Flusher)
    if !ok {
        s.writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "streaming not supported"})
        return errors.New("streaming not supported")
    }

    for {
        select {
        case <-ctx.Done():
            _ = writeSSE(w, map[string]any{"type": "error", "data": "timeout"})
            flusher.Flush()
            return ctx.Err()
        case event, ok := <-ch:
            if !ok {
                return nil
            }
            if event.Type == "tool_end" {
                s.logToolError(ctx, event)
            }
            if event.Type == "error" {
                s.logStreamError(ctx, "events", event.Data)
            }
            if err := writeSSE(w, event); err != nil {
                return err
            }
            flusher.Flush()
        }
    }
}

func (s *Server) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
    if s.opts.RequestTimeout <= 0 {
        return ctx, func() {}
    }
    return context.WithTimeout(ctx, s.opts.RequestTimeout)
}

func (s *Server) readJSON(w http.ResponseWriter, r *http.Request, dst any) error {
    r.Body = http.MaxBytesReader(w, r.Body, s.opts.MaxBodyBytes)
    defer r.Body.Close()

    decoder := json.NewDecoder(r.Body)
    if err := decoder.Decode(dst); err != nil {
        return err
    }
    return nil
}

func (s *Server) writeRunError(w http.ResponseWriter, err error) {
    if errors.Is(err, kazi.ErrNotImplemented) {
        s.writeJSON(w, http.StatusNotImplemented, ErrorResponse{Error: "not implemented"})
        return
    }
    if errors.Is(err, kazi.ErrNoLLM) {
        s.writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "llm not configured"})
        return
    }
    if errors.Is(err, context.DeadlineExceeded) {
        s.writeJSON(w, http.StatusGatewayTimeout, ErrorResponse{Error: "request timed out"})
        return
    }
    s.writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "request failed"})
}

func (s *Server) isAuthed(r *http.Request) bool {
    if s.opts.APIKey == "" {
        return true
    }
    auth := r.Header.Get("Authorization")
    if !strings.HasPrefix(auth, "Bearer ") {
        return false
    }
    token := strings.TrimPrefix(auth, "Bearer ")
    return subtle.ConstantTimeCompare([]byte(token), []byte(s.opts.APIKey)) == 1
}

func (s *Server) writeJSON(w http.ResponseWriter, status int, payload any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(payload)
}

func writeSSE(w http.ResponseWriter, payload any) error {
    data, err := json.Marshal(payload)
    if err != nil {
        return err
    }
    _, err = fmt.Fprintf(w, "data: %s\n\n", data)
    return err
}

func (s *Server) recordRequest(name string) {
    metric := s.metrics[name]
    if metric == nil {
        return
    }
    metric.count.Add(1)
}

func (s *Server) recordError(name string) {
    metric := s.metrics[name]
    if metric == nil {
        return
    }
    metric.errors.Add(1)
}

func (s *Server) recordLatency(name string, duration time.Duration) {
    metric := s.metrics[name]
    if metric == nil {
        return
    }
    metric.latencyMs.Add(duration.Milliseconds())
}

func (s *Server) observeProm(route, method string, status int, duration time.Duration) {
    if s.prom == nil {
        return
    }
    s.prom.Observe(route, method, status, duration)
}

func (s *Server) setActiveGauge() {
    if s.prom == nil {
        return
    }
    s.prom.SetActive(s.activeRequests.Load())
}

func (s *Server) logRequest(route string, r *http.Request, sw *statusWriter, requestID string, ip string, duration time.Duration) {
    fields := map[string]any{
        "route":        route,
        "method":       r.Method,
        "path":         r.URL.Path,
        "status":       sw.Status(),
        "duration_ms":  duration.Milliseconds(),
        "request_id":   requestID,
        "client_ip":    ip,
        "user_agent":   r.UserAgent(),
    }
    s.logger.Info("request", fields)
}

func (s *Server) logToolError(ctx context.Context, event kazi.StreamEvent) {
    if s.logger == nil {
        return
    }
    data, ok := event.Data.(map[string]any)
    if !ok {
        return
    }
    errValue, hasError := data["error"]
    if !hasError || errValue == nil {
        return
    }
    fields := map[string]any{
        "request_id": requestIDFromContext(ctx),
        "tool_name":  data["name"],
        "tool_id":    data["id"],
        "error":      errValue,
    }
    s.logger.Error("tool error", fields)
}

func (s *Server) logStreamError(ctx context.Context, route string, data any) {
    if s.logger == nil {
        return
    }
    fields := map[string]any{
        "route":      route,
        "request_id": requestIDFromContext(ctx),
        "error":      data,
    }
    s.logger.Error("stream error", fields)
}

type statusWriter struct {
    http.ResponseWriter
    status int
    bytes  int
}

func newStatusWriter(w http.ResponseWriter) *statusWriter {
    return &statusWriter{ResponseWriter: w}
}

func (w *statusWriter) WriteHeader(statusCode int) {
    w.status = statusCode
    w.ResponseWriter.WriteHeader(statusCode)
}

func (w *statusWriter) Write(data []byte) (int, error) {
    if w.status == 0 {
        w.status = http.StatusOK
    }
    n, err := w.ResponseWriter.Write(data)
    w.bytes += n
    return n, err
}

func (w *statusWriter) Status() int {
    if w.status == 0 {
        return http.StatusOK
    }
    return w.status
}

func (w *statusWriter) Flush() {
    if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
        flusher.Flush()
    }
}

type requestIDKey struct{}

func withRequestID(ctx context.Context, requestID string) context.Context {
    return context.WithValue(ctx, requestIDKey{}, requestID)
}

func requestIDFromContext(ctx context.Context) string {
    value := ctx.Value(requestIDKey{})
    if id, ok := value.(string); ok {
        return id
    }
    return ""
}

func (s *Server) shouldSample() bool {
    if s.opts.LogSampleRate >= 1 {
        return true
    }
    if s.opts.LogSampleRate <= 0 {
        return false
    }
    buf := make([]byte, 8)
    if _, err := rand.Read(buf); err != nil {
        return true
    }
    value := binary.LittleEndian.Uint64(buf)
    ratio := float64(value) / float64(^uint64(0))
    return ratio <= s.opts.LogSampleRate
}

func (s *Server) applySecurityHeaders(w http.ResponseWriter) {
    w.Header().Set("X-Content-Type-Options", "nosniff")
    w.Header().Set("X-Frame-Options", "DENY")
    w.Header().Set("X-XSS-Protection", "1; mode=block")
    w.Header().Set("Referrer-Policy", "no-referrer")
    w.Header().Set("Permissions-Policy", "interest-cohort=()")
    w.Header().Set("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
}

func (s *Server) ensureRequestID(w http.ResponseWriter, r *http.Request) string {
    raw := r.Header.Get("X-Request-ID")
    if raw == "" || !isSafeRequestID(raw) {
        raw = newRequestID()
    }
    w.Header().Set("X-Request-ID", raw)
    return raw
}

func isSafeRequestID(value string) bool {
    if len(value) == 0 || len(value) > 64 {
        return false
    }
    for _, r := range value {
        if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' {
            continue
        }
        return false
    }
    return true
}

func newRequestID() string {
    buf := make([]byte, 16)
    if _, err := rand.Read(buf); err != nil {
        return fmt.Sprintf("%d", time.Now().UnixNano())
    }
    return hex.EncodeToString(buf)
}

func (s *Server) clientIP(r *http.Request) string {
    host, _, err := net.SplitHostPort(r.RemoteAddr)
    if err != nil {
        return r.RemoteAddr
    }
    return host
}

func (s *Server) checkRateLimit(ip string) bool {
    if s.opts.RateLimitPerMinute <= 0 {
        return true
    }
    now := time.Now()
    s.rateMu.Lock()
    defer s.rateMu.Unlock()
    entry, ok := s.rate[ip]
    if !ok || now.Sub(entry.window) >= time.Minute {
        s.rate[ip] = &rateEntry{window: now, count: 1}
        return true
    }
    if entry.count >= s.opts.RateLimitPerMinute {
        return false
    }
    entry.count += 1
    return true
}

func sanitizeRunRequest(req *RunRequest) {
    req.ThreadID = sanitizeThreadID(req.ThreadID)
    req.UserID = sanitizeString(req.UserID, 256)
    req.TenantID = sanitizeString(req.TenantID, 256)
}

func sanitizeThreadID(value string) string {
    value = sanitizeString(value, 256)
    if value == "" {
        return value
    }
    var b strings.Builder
    for _, r := range value {
        if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' || r == ':' || r == '.' || r == '@' {
            b.WriteRune(r)
        } else {
            b.WriteRune('_')
        }
    }
    return b.String()
}

func sanitizeString(value string, max int) string {
    value = strings.ReplaceAll(value, "\x00", "")
    if max > 0 && len(value) > max {
        return value[:max]
    }
    return value
}

func buildRunOptions(req RunRequest) []kazi.RunOption {
    opts := make([]kazi.RunOption, 0)
    if req.ThreadID != "" {
        opts = append(opts, kazi.WithThreadID(req.ThreadID))
    }
    if req.UserID != "" {
        opts = append(opts, kazi.WithUserID(req.UserID))
    }
    if req.TenantID != "" {
        opts = append(opts, kazi.WithTenantID(req.TenantID))
    }
    if req.SystemPrompt != "" {
        opts = append(opts, kazi.WithSystemPrompt(req.SystemPrompt))
    }
    if req.MaxToolCalls > 0 {
        opts = append(opts, kazi.WithMaxToolCalls(req.MaxToolCalls))
    }
    if req.TrackCost {
        opts = append(opts, kazi.WithTrackCost(req.TrackCost))
    }
    return opts
}
