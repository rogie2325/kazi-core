package secrets

import (
    "encoding"
    "os"
)

type SecretResolver func() (string, bool)

type SecretRef struct {
    value    *string
    resolver SecretResolver
}

var _ encoding.TextUnmarshaler = (*SecretRef)(nil)
var _ encoding.TextMarshaler = (*SecretRef)(nil)

func FromLiteral(value string) SecretRef {
    v := value
    return SecretRef{value: &v}
}

func FromEnv(name string, fallback string) SecretRef {
    return SecretRef{
        resolver: func() (string, bool) {
            if v, ok := os.LookupEnv(name); ok {
                return v, true
            }
            if fallback != "" {
                return fallback, true
            }
            return "", false
        },
    }
}

func FromFunc(fn SecretResolver) SecretRef {
    return SecretRef{resolver: fn}
}

func (s SecretRef) Resolve() (string, bool) {
    if s.value != nil {
        return *s.value, true
    }
    if s.resolver != nil {
        return s.resolver()
    }
    return "", false
}

func (s *SecretRef) UnmarshalText(text []byte) error {
    v := string(text)
    s.value = &v
    s.resolver = nil
    return nil
}

func (s SecretRef) MarshalText() ([]byte, error) {
    return []byte("***"), nil
}

func (s SecretRef) String() string {
    return "***"
}

func (s SecretRef) GoString() string {
    return "SecretRef(***)"
}
