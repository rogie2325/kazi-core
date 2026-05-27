package config

import (
    "encoding/json"

    "github.com/invopop/jsonschema"
)

func Schema() *jsonschema.Schema {
    reflector := jsonschema.Reflector{
        AllowAdditionalProperties: false,
    }
    return reflector.Reflect(Config{})
}

func SchemaJSON() ([]byte, error) {
    return json.MarshalIndent(Schema(), "", "  ")
}
