package config

import (
    "io"
    "os"

    "gopkg.in/yaml.v3"
)

func LoadFile(path string) (Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return Config{}, err
    }
    return LoadBytes(data)
}

func Load(r io.Reader) (Config, error) {
    data, err := io.ReadAll(r)
    if err != nil {
        return Config{}, err
    }
    return LoadBytes(data)
}

func LoadBytes(data []byte) (Config, error) {
    cfg := DefaultConfig()
    if len(data) == 0 {
        return cfg, nil
    }
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return Config{}, err
    }
    return cfg, nil
}
