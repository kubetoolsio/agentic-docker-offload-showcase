# AI Docker Offload Demo - Makefile

.PHONY: help setup install validate start stop restart logs test test-quick test-load clean gpu-test docs

# Default target
help:
	@echo "🚀 AI Docker Offload Demo - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "📋 Setup & Installation:"
	@echo "  make install       - Complete setup (validate, setup, download models)"
	@echo "  make setup         - Run setup script"
	@echo "  make validate      - Validate system prerequisites"
	@echo "  make models        - Download AI models"
	@echo ""
	@echo "🐳 Docker Operations:"
	@echo "  make start         - Start all services"
	@echo "  make stop          - Stop all services"
	@echo "  make restart       - Restart all services"
	@echo "  make logs          - View service logs"
	@echo "  make ps            - Show running containers"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test          - Run comprehensive tests"
	@echo "  make test-quick    - Run quick health checks"
	@echo "  make test-load     - Run load testing"
	@echo "  make test-gpu      - Test GPU functionality"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  make monitor       - Start monitoring stack"
	@echo "  make metrics       - Show system metrics"
	@echo "  make dashboard     - Open Grafana dashboard"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean         - Clean up containers and volumes"
	@echo "  make clean-all     - Clean everything including images"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs          - Generate documentation"
	@echo "  make readme        - View README"

# Installation and setup
install: validate setup models
	@echo "✅ Installation complete!"
	@echo "💡 Next steps:"
	@echo "   make start     - Start the system"
	@echo "   make test      - Run tests"

validate:
	@echo "🔍 Validating system prerequisites..."
	@chmod +x scripts/validate-system.sh
	@./scripts/validate-system.sh

setup:
	@echo "📁 Setting up project structure..."
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

models:
	@echo "📥 Downloading AI models..."
	@chmod +x scripts/download-models.sh
	@./scripts/download-models.sh

# Docker operations
start:
	@echo "🚀 Starting AI Docker Offload system..."
	@docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@make ps
	@echo ""
	@echo "🎉 System started successfully!"
	@echo "📊 Access points:"
	@echo "   • API: http://localhost:8080"
	@echo "   • Triton: http://localhost:8000"
	@echo "   • Health: http://localhost:8080/health"

start-gpu:
	@echo "🚀 Starting with GPU offload support..."
	@docker-compose --profile gpu-offload up -d

start-monitoring:
	@echo "📊 Starting with monitoring stack..."
	@docker-compose --profile monitoring up -d

stop:
	@echo "🛑 Stopping AI Docker Offload system..."
	@docker-compose down

restart: stop start

logs:
	@echo "📋 Viewing service logs..."
	@docker-compose logs -f --tail=50

logs-coordinator:
	@docker-compose logs -f inference-coordinator

logs-triton:
	@docker-compose logs -f triton-server

logs-preprocessor:
	@docker-compose logs -f preprocessor

ps:
	@echo "📋 Container status:"
	@docker-compose ps

# Testing
test:
	@echo "🧪 Running comprehensive test suite..."
	@chmod +x scripts/test-complete.sh
	@./scripts/test-complete.sh

test-quick:
	@echo "⚡ Running quick health checks..."
	@chmod +x scripts/test-complete.sh
	@./scripts/test-complete.sh quick

test-system:
	@echo "🔧 Running system tests..."
	@chmod +x scripts/test-system.sh
	@./scripts/test-system.sh

test-inference:
	@echo "🤖 Running inference tests..."
	@chmod +x scripts/test-inference.sh
	@./scripts/test-inference.sh

test-load:
	@echo "📈 Running load tests..."
	@chmod +x scripts/load-test.sh
	@./scripts/load-test.sh

test-load-stress:
	@echo "💪 Running stress tests..."
	@chmod +x scripts/load-test.sh
	@./scripts/load-test.sh stress

test-gpu:
	@echo "🖥️  Running GPU tests..."
	@chmod +x scripts/test-complete.sh
	@./scripts/test-complete.sh gpu

benchmark:
	@echo "⚡ Running performance benchmarks..."
	@chmod +x scripts/benchmark.sh
	@./scripts/benchmark.sh

# Monitoring and metrics
monitor:
	@echo "📊 Starting monitoring stack..."
	@docker-compose --profile monitoring up -d prometheus grafana
	@echo "📊 Monitoring available at:"
	@echo "   • Prometheus: http://localhost:9090"
	@echo "   • Grafana: http://localhost:3000 (admin/admin)"

metrics:
	@echo "📈 System metrics:"
	@echo "Coordinator metrics:"
	@curl -s http://localhost:8080/metrics | grep -E "(inference_requests|http_requests)" | head -5 || echo "Metrics not available"
	@echo ""
	@echo "Triton metrics:"
	@curl -s http://localhost:8000/metrics | grep -E "(nv_inference|gpu)" | head -5 || echo "Metrics not available"

dashboard:
	@echo "📊 Opening Grafana dashboard..."
	@echo "URL: http://localhost:3000"
	@echo "Login: admin/admin"
	@command -v open >/dev/null 2>&1 && open "http://localhost:3000" || \
	 command -v xdg-open >/dev/null 2>&1 && xdg-open "http://localhost:3000" || \
	 echo "Please open http://localhost:3000 in your browser"

# Health checks
health:
	@echo "🏥 System health check:"
	@echo "Coordinator:"
	@curl -s http://localhost:8080/health | jq . || echo "❌ Not available"
	@echo "Triton:"
	@curl -s http://localhost:8000/v2/health/ready | jq . || echo "❌ Not available"

status:
	@echo "📊 System status:"
	@make health
	@echo ""
	@make ps

# Development helpers
shell-coordinator:
	@docker-compose exec inference-coordinator /bin/bash

shell-triton:
	@docker-compose exec triton-server /bin/bash

shell-preprocessor:
	@docker-compose exec preprocessor /bin/bash

# Scaling operations
scale-preprocessor:
	@echo "📈 Scaling preprocessor to 3 instances..."
	@docker-compose up -d --scale preprocessor=3

scale-down:
	@echo "📉 Scaling back to single instances..."
	@docker-compose up -d --scale preprocessor=1

# Cleanup
clean:
	@echo "🧹 Cleaning up containers and volumes..."
	@docker-compose down -v
	@docker system prune -f

clean-all: clean
	@echo "🧹 Cleaning up everything including images..."
	@docker-compose down -v --rmi all
	@docker system prune -af

clean-models:
	@echo "🧹 Cleaning downloaded models..."
	@rm -rf models/
	@rm -rf test-data/

reset: clean-all
	@echo "🔄 Resetting system to initial state..."
	@rm -rf models/ test-data/ monitoring/ logs/
	@echo "✅ System reset complete. Run 'make install' to set up again."

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "# AI Docker Offload Documentation" > docs.md
	@echo "Generated: $(shell date)" >> docs.md
	@echo "" >> docs.md
	@docker-compose config >> docs.md

readme:
	@cat README.md

# Environment setup
env:
	@echo "🔧 Creating environment file..."
	@cp .env.example .env
	@echo "✅ Created .env file. Please review and customize as needed."

# Quick shortcuts
up: start
down: stop
build:
	@docker-compose build --no-cache

rebuild: clean build start

# Development workflow
dev: validate setup models start test-quick
	@echo "🎉 Development environment ready!"
	@echo "💡 Run 'make test' for full testing"

# Production workflow  
prod: validate setup models start-monitoring test
	@echo "🚀 Production deployment complete!"
	@echo "📊 Monitoring: http://localhost:3000"

# CI/CD helpers
ci-test: start test stop

# Info commands
info:
	@echo "📋 System Information:"
	@echo "Docker version: $(shell docker --version)"
	@echo "Docker Compose version: $(shell docker-compose --version || docker compose version)"
	@echo "Available GPU: $(shell nvidia-smi -L 2>/dev/null | wc -l || echo '0') NVIDIA GPU(s)"
	@echo "System memory: $(shell free -h | awk '/^Mem:/{print $$2}' || echo 'Unknown')"
	@echo "Available disk: $(shell df -h / | awk 'NR==2{print $$4}' || echo 'Unknown')"

version:
	@echo "AI Docker Offload Demo v1.0.0"
	@echo "Components:"
	@echo "  • NVIDIA Triton Server: 24.01"
	@echo "  • Docker Compose: $(shell docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)"
	@echo "  • Python Agents: 3.9+"