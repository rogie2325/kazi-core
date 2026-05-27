package serve

import (
    "encoding/json"
    "io"
    "log"
    "time"
)

type Logger interface {
    Info(msg string, fields map[string]any)
    Error(msg string, fields map[string]any)
}

type JSONLogger struct {
    logger *log.Logger
}

func NewJSONLogger(w io.Writer) *JSONLogger {
    return &JSONLogger{logger: log.New(w, "", 0)}
}

func (l *JSONLogger) Info(msg string, fields map[string]any) {
    l.write("info", msg, fields)
}

func (l *JSONLogger) Error(msg string, fields map[string]any) {
    l.write("error", msg, fields)
}

func (l *JSONLogger) write(level string, msg string, fields map[string]any) {
    payload := map[string]any{
        "level": level,
        "msg":   msg,
        "ts":    time.Now().UTC().Format(time.RFC3339Nano),
    }
    for key, value := range fields {
        payload[key] = value
    }
    data, err := json.Marshal(payload)
    if err != nil {
        l.logger.Printf("{\"level\":\"error\",\"msg\":\"log marshal failed\",\"err\":%q}", err.Error())
        return
    }
    l.logger.Println(string(data))
}
