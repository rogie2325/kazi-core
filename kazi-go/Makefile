SHELL := /bin/sh

MODULE := github.com/rogie2325/kazi
BINARY := bin/kazi

.PHONY: help build test vet fmt lint tidy cover clean run serve validate config-schema

help:
	@echo "Targets:"
	@echo "  build          Build the CLI"
	@echo "  test           Run tests"
	@echo "  vet            Run go vet"
	@echo "  fmt            Run gofmt"
	@echo "  lint           Run fmt + vet"
	@echo "  tidy           Run go mod tidy"
	@echo "  cover          Run tests with coverage"
	@echo "  run            Run CLI with ARGS=..."
	@echo "  serve          Start server with ARGS=..."
	@echo "  validate       Run CLI validate with CONFIG=..."
	@echo "  config-schema  Print config schema"
	@echo "  clean          Remove build artifacts"

build:
	@mkdir -p bin
	go build -o $(BINARY) ./cmd/kazi

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

lint: fmt vet

tidy:
	go mod tidy

cover:
	go test ./... -coverprofile=coverage.out
	go tool cover -html=coverage.out -o coverage.html

run:
	go run ./cmd/kazi $(ARGS)

serve:
	go run ./cmd/kazi serve $(ARGS)

validate:
	@if [ -z "$(CONFIG)" ]; then echo "CONFIG is required"; exit 1; fi
	go run ./cmd/kazi validate $(CONFIG)

config-schema:
	go run ./cmd/kazi config-schema

clean:
	rm -rf bin coverage.out coverage.html
