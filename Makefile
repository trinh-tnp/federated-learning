SHELL := /bin/bash

PHONY: help
help:
	@echo "Available commands:"
	@echo "  export          Use \$$\\(make export\\) to set virtualenv path"
	@echo "  activate        use \$$\\(make activate\\) to activate the virtualenv"

export:
	@echo "export UV_PROJECT_ENVIRONMENT=./.wsl-venv"

activate:
	@echo "source ./.wsl-venv/bin/activate"

run:
	@echo "Running Flower with the specified configuration..."
	@flwr run .