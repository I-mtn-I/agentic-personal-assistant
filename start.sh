#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$PROJECT_ROOT/Docker"
VOLUMES_BASE="$PROJECT_ROOT/.docker-volumes"

echo "Starting Agentic Personal Assistant..."

echo "Creating required directories..."
mkdir -p "$VOLUMES_BASE/ollama-data"
mkdir -p "$VOLUMES_BASE/postgres-data"
mkdir -p "$VOLUMES_BASE/qdrant-data"
echo " Directories created"

cd "$DOCKER_DIR"

echo "Starting Docker Compose services..."
docker compose up -d --remove-orphans
echo " Docker services started"

echo "Waiting for Ollama service to be ready..."
sleep 5

LLM_MODEL=$(grep "^LLM_MODEL=" "$PROJECT_ROOT/services/.env" | cut -d'=' -f2)
echo "Downloading LLM model: $LLM_MODEL"
docker exec ollama ollama pull "$LLM_MODEL"
echo " Model downloaded"

echo ""
echo "All services started successfully!"
echo "-------------------------------------"
echo "Available services:"
echo "  - Ollama:     http://localhost:11434"
echo "  - PostgreSQL: localhost:5434"
echo "  - Qdrant:     http://localhost:6333"
echo "-------------------------------------"
