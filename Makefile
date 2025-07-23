.PHONY: help install test lint format clean docker-build docker-run k8s-deploy backup

PYTHON := python3.9
PIP := pip3
VENV := venv
PROJECT_NAME := quantum-trading-system
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "latest")
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
NAMESPACE := quantum-trading

help:
	@echo "Quantum Trading System - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install          Install dependencies and setup environment"
	@echo "  test             Run test suite"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black and isort"
	@echo "  clean            Clean build artifacts and cache"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run system in Docker"
	@echo "  docker-compose   Start all services with docker-compose"
	@echo "  docker-stop      Stop all Docker services"
	@echo ""
	@echo "Kubernetes:"
	@echo "  k8s-deploy       Deploy to Kubernetes"
	@echo "  k8s-status       Check Kubernetes deployment status"
	@echo "  k8s-logs         View application logs"
	@echo "  k8s-shell        Open shell in running pod"
	@echo "  k8s-delete       Delete Kubernetes deployment"
	@echo ""
	@echo "Operations:"
	@echo "  backup           Create system backup"
	@echo "  restore          Restore from backup"
	@echo "  migrate          Run database migrations"
	@echo "  monitoring       Setup monitoring stack"

install:
	@echo "Setting up Quantum Trading System..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools wheel
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -r requirements-dev.txt
	@echo "Installing pre-commit hooks..."
	$(VENV)/bin/pre-commit install
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

test:
	@echo "Running test suite..."
	$(VENV)/bin/python -m pytest tests/ -v --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	$(VENV)/bin/python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

test-integration:
	@echo "Running integration tests..."
	$(VENV)/bin/python -m pytest tests/integration/ -v --tb=short

test-performance:
	@echo "Running performance tests..."
	$(VENV)/bin/python -m pytest tests/performance/ -v --tb=short

lint:
	@echo "Running code linting..."
	$(VENV)/bin/flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(VENV)/bin/flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	$(VENV)/bin/mypy . --ignore-missing-imports

format:
	@echo "Formatting code..."
	$(VENV)/bin/black .
	$(VENV)/bin/isort .

clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .
	docker tag $(DOCKER_IMAGE) $(PROJECT_NAME):latest

docker-run:
	@echo "Running Docker container..."
	docker run -d \
		--name quantum-trading \
		--env-file .env.production \
		-p 8080:8080 \
		-p 8050:8050 \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE)

docker-compose:
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "Services started. Dashboard: http://localhost:8050"

docker-stop:
	@echo "Stopping Docker services..."
	docker-compose down
	docker stop quantum-trading || true
	docker rm quantum-trading || true

docker-logs:
	@echo "Following Docker logs..."
	docker-compose logs -f quantum-trading-system

k8s-deploy:
	@echo "Deploying to Kubernetes..."
	./scripts/deploy.sh production deploy

k8s-status:
	@echo "Checking Kubernetes deployment status..."
	kubectl get all -n $(NAMESPACE)
	kubectl get pv,pvc -n $(NAMESPACE)

k8s-logs:
	@echo "Viewing application logs..."
	kubectl logs -f deployment/quantum-trading -n $(NAMESPACE)

k8s-shell:
	@echo "Opening shell in running pod..."
	kubectl exec -it deployment/quantum-trading -n $(NAMESPACE) -- /bin/bash

k8s-delete:
	@echo "Deleting Kubernetes deployment..."
	kubectl delete namespace $(NAMESPACE)

k8s-restart:
	@echo "Restarting Kubernetes deployment..."
	kubectl rollout restart deployment/quantum-trading -n $(NAMESPACE)

k8s-scale:
	@echo "Scaling deployment to $(REPLICAS) replicas..."
	kubectl scale deployment quantum-trading --replicas=$(REPLICAS) -n $(NAMESPACE)

backup:
	@echo "Creating system backup..."
	./scripts/deploy.sh production backup

restore:
	@echo "Restoring from backup..."
	@read -p "Enter backup directory: " BACKUP_DIR; \
	kubectl exec deployment/quantum-postgres -n $(NAMESPACE) -- psql -U quantum -d quantum_trading < $$BACKUP_DIR/database-backup.sql

migrate:
	@echo "Running database migrations..."
	kubectl exec -it deployment/quantum-postgres -n $(NAMESPACE) -- psql -U quantum -d quantum_trading -f /docker-entrypoint-initdb.d/init.sql

monitoring:
	@echo "Setting up monitoring stack..."
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo update
	helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
		--namespace monitoring --create-namespace \
		--values monitoring/prometheus-values.yaml
	helm upgrade --install grafana grafana/grafana \
		--namespace monitoring \
		--values monitoring/grafana-values.yaml

setup-dev:
	@echo "Setting up development environment..."
	./scripts/setup.sh
	make install
	@echo "Development environment ready!"

run-local:
	@echo "Starting local development server..."
	$(VENV)/bin/python -m quantum_trading_system

run-dashboard:
	@echo "Starting dashboard..."
	$(VENV)/bin/streamlit run monitoring/dashboard.py --server.port 8050

run-backtest:
	@echo "Running backtest..."
	$(VENV)/bin/python -c "
import asyncio
from backtesting_engine import AdvancedBacktestingEngine, BacktestConfig, sample_momentum_strategy
from datetime import datetime

async def main():
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        symbols=['BTC-USD', 'ETH-USD'],
        benchmark='SPY',
        risk_free_rate=0.02,
        rebalance_frequency='daily',
        position_sizing='equal_weight',
        max_position_size=0.5,
        stop_loss=None,
        take_profit=None
    )
    
    engine = AdvancedBacktestingEngine(config)
    await engine.load_historical_data(['BTC-USD', 'ETH-USD'])
    result = await engine.run_backtest(sample_momentum_strategy, 'Momentum Strategy')
    
    print(f'Total Return: {result.total_return:.2%}')
    print(f'Sharpe Ratio: {result.sharpe_ratio:.2f}')
    print(f'Max Drawdown: {result.max_drawdown:.2%}')

asyncio.run(main())
"

security-scan:
	@echo "Running security scan..."
	$(VENV)/bin/bandit -r . -x tests/
	$(VENV)/bin/safety check

performance-test:
	@echo "Running performance benchmarks..."
	$(VENV)/bin/python -m pytest tests/performance/ --benchmark-only

stress-test:
	@echo "Running stress tests..."
	$(VENV)/bin/python tests/stress_test.py

health-check:
	@echo "Running health checks..."
	curl -f http://localhost:8080/health || echo "Service not running"
	curl -f http://localhost:8050/health || echo "Dashboard not running"

update-deps:
	@echo "Updating dependencies..."
	$(VENV)/bin/pip list --outdated --format=json | jq -r '.[] | .name' | xargs -I {} $(VENV)/bin/pip install --upgrade {}
	$(VENV)/bin/pip freeze > requirements.txt

generate-docs:
	@echo "Generating documentation..."
	$(VENV)/bin/sphinx-build -b html docs/ docs/_build/

release:
	@echo "Creating release $(VERSION)..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker push $(PROJECT_NAME):$(VERSION)

init-secrets:
	@echo "Initializing secrets..."
	kubectl create secret generic quantum-secrets \
		--from-env-file=.env.production \
		--namespace=$(NAMESPACE) \
		--dry-run=client -o yaml | kubectl apply -f -

port-forward:
	@echo "Setting up port forwarding..."
	kubectl port-forward service/quantum-trading-service 8080:80 -n $(NAMESPACE) &
	kubectl port-forward service/quantum-grafana 3000:3000 -n monitoring &
	@echo "Services available at:"
	@echo "  Trading API: http://localhost:8080"
	@echo "  Grafana: http://localhost:3000"