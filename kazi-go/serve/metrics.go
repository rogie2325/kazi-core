package serve

import (
    "net/http"
    "strconv"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

type promMetrics struct {
    registry        *prometheus.Registry
    requestsTotal   *prometheus.CounterVec
    requestErrors   *prometheus.CounterVec
    requestDuration *prometheus.HistogramVec
    activeRequests  prometheus.Gauge
    toolsRegistered prometheus.Gauge
    toolCallsTotal  prometheus.Gauge
    toolErrorsTotal prometheus.Gauge
}

func newPromMetrics() *promMetrics {
    reg := prometheus.NewRegistry()
    m := &promMetrics{
        registry: reg,
        requestsTotal: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "kazi_http_requests_total",
                Help: "Total HTTP requests by route, method, and status.",
            },
            []string{"route", "method", "status"},
        ),
        requestErrors: prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "kazi_http_request_errors_total",
                Help: "Total HTTP request errors by route, method, and status.",
            },
            []string{"route", "method", "status"},
        ),
        requestDuration: prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "kazi_http_request_duration_seconds",
                Help:    "HTTP request duration in seconds.",
                Buckets: prometheus.DefBuckets,
            },
            []string{"route", "method"},
        ),
        activeRequests: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "kazi_http_active_requests",
            Help: "Number of in-flight HTTP requests.",
        }),
        toolsRegistered: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "kazi_tools_registered",
            Help: "Number of tools currently registered.",
        }),
        toolCallsTotal: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "kazi_tool_calls_total",
            Help: "Total tool calls executed.",
        }),
        toolErrorsTotal: prometheus.NewGauge(prometheus.GaugeOpts{
            Name: "kazi_tool_errors_total",
            Help: "Total tool call errors.",
        }),
    }

    reg.MustRegister(
        m.requestsTotal,
        m.requestErrors,
        m.requestDuration,
        m.activeRequests,
        m.toolsRegistered,
        m.toolCallsTotal,
        m.toolErrorsTotal,
    )

    return m
}

func (m *promMetrics) Handler() http.Handler {
    return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

func (m *promMetrics) Observe(route, method string, status int, duration time.Duration) {
    if m == nil {
        return
    }
    statusLabel := strconv.Itoa(status)
    m.requestsTotal.WithLabelValues(route, method, statusLabel).Inc()
    m.requestDuration.WithLabelValues(route, method).Observe(duration.Seconds())
    if status >= 400 {
        m.requestErrors.WithLabelValues(route, method, statusLabel).Inc()
    }
}

func (m *promMetrics) SetActive(count int64) {
    if m == nil {
        return
    }
    m.activeRequests.Set(float64(count))
}

func (m *promMetrics) SetToolStats(registered int, calls int64, errors int64) {
    if m == nil {
        return
    }
    m.toolsRegistered.Set(float64(registered))
    m.toolCallsTotal.Set(float64(calls))
    m.toolErrorsTotal.Set(float64(errors))
}
