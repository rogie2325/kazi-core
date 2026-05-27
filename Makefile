SHELL := /bin/sh

GO_DIR := kazi-go

.PHONY: help \
	go-build go-test go-vet go-fmt go-lint go-tidy go-cover go-run go-serve go-validate go-config-schema go-clean \
	py-install py-install-uv py-test py-test-unit py-test-integration py-lint py-lint-fix py-typecheck py-security py-validate py-config-schema

help:
	@echo "Go targets (kazi-go):"
	@echo "  go-build          Build the CLI"
	@echo "  go-test           Run tests"
	@echo "  go-vet            Run go vet"
	@echo "  go-fmt            Run gofmt"
	@echo "  go-lint           Run fmt + vet"
	@echo "  go-tidy           Run go mod tidy"
	@echo "  go-cover          Run tests with coverage"
	@echo "  go-run            Run CLI with ARGS=..."
	@echo "  go-serve          Start server with ARGS=..."
	@echo "  go-validate       Run CLI validate with CONFIG=..."
	@echo "  go-config-schema  Print config schema"
	@echo "  go-clean          Remove Go build artifacts"
	@echo ""
	@echo "Python targets (root):"
	@echo "  py-install        pip install -e .[dev]"
	@echo "  py-install-uv     uv sync"
	@echo "  py-test           Run all tests"
	@echo "  py-test-unit      Run unit tests"
	@echo "  py-test-integration Run integration tests"
	@echo "  py-lint           Run ruff check"
	@echo "  py-lint-fix       Run ruff check --fix"
	@echo "  py-typecheck      Run mypy"
	@echo "  py-security       Run bandit"
	@echo "  py-validate       Run python -m kazi validate CONFIG=..."
	@echo "  py-config-schema  Print Python config schema"

# Go targets

go-build:
	$(MAKE) -C $(GO_DIR) build

go-test:
	$(MAKE) -C $(GO_DIR) test

go-vet:
	$(MAKE) -C $(GO_DIR) vet

go-fmt:
	$(MAKE) -C $(GO_DIR) fmt

go-lint:
	$(MAKE) -C $(GO_DIR) lint

go-tidy:
	$(MAKE) -C $(GO_DIR) tidy

go-cover:
	$(MAKE) -C $(GO_DIR) cover

go-run:
	$(MAKE) -C $(GO_DIR) run ARGS="$(ARGS)"

go-serve:
	$(MAKE) -C $(GO_DIR) serve ARGS="$(ARGS)"

go-validate:
	$(MAKE) -C $(GO_DIR) validate CONFIG="$(CONFIG)"

go-config-schema:
	$(MAKE) -C $(GO_DIR) config-schema

go-clean:
	$(MAKE) -C $(GO_DIR) clean

# Python targets

py-install:
	pip install -e ".[dev]"

py-install-uv:
	uv sync

py-test:
	pytest

py-test-unit:
	pytest tests/unit/

py-test-integration:
	pytest tests/integration/

py-lint:
	ruff check kazi/ tests/

py-lint-fix:
	ruff check --fix kazi/ tests/

py-typecheck:
	mypy kazi/ --ignore-missing-imports

py-security:
	bandit -r kazi/ -ll --skip B603,B607

py-validate:
	@if [ -z "$(CONFIG)" ]; then echo "CONFIG is required"; exit 1; fi
	python -m kazi validate $(CONFIG)

py-config-schema:
	python -m kazi config-schema
