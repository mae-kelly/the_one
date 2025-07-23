#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Setting up Quantum Trading System..."

install_system_dependencies() {
    echo "📦 Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            python3.9 \
            python3.9-dev \
            python3-pip \
            build-essential \
            libssl-dev \
            libffi-dev \
            libpq-dev \
            redis-server \
            postgresql-client \
            curl \
            wget \
            git \
            vim \
            htop \
            screen \
            tmux
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew &> /dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        brew update
        brew install \
            python@3.9 \
            postgresql \
            redis \
            curl \
            wget \
            git \
            vim \
            htop \
            tmux
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    echo "✅ System dependencies installed"
}

setup_python_environment() {
    echo "🐍 Setting up Python environment..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -d "venv" ]]; then
        python3.9 -m venv venv
    fi
    
    source venv/bin/activate
    
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    
    echo "✅ Python environment setup complete"
}

setup_cuda() {
    echo "🚀 Setting up CUDA environment..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected"
        
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | head -1)
        echo "CUDA Version: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == "11."* ]]; then
            pip install cupy-cuda11x
        elif [[ "$CUDA_VERSION" == "12."* ]]; then
            pip install cupy-cuda12x
        else
            echo "⚠️  Unsupported CUDA version: $CUDA_VERSION"
            echo "Installing CPU-only version"
        fi
        
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️  No NVIDIA GPU detected, installing CPU-only versions"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    echo "✅ CUDA environment setup complete"
}

setup_databases() {
    echo "🗄️  Setting up databases..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start postgresql
        sudo systemctl start redis-server
        sudo systemctl enable postgresql
        sudo systemctl enable redis-server
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew services start postgresql
        brew services start redis
    fi
    
    sleep 5
    
    sudo -u postgres psql -c "CREATE DATABASE quantum_trading;" || echo "Database already exists"
    sudo -u postgres psql -c "CREATE USER quantum WITH PASSWORD 'quantum_pass';" || echo "User already exists"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE quantum_trading TO quantum;" || echo "Privileges already granted"
    
    echo "✅ Databases setup complete"
}

setup_configuration() {
    echo "⚙️  Setting up configuration..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f ".env.production" ]]; then
        cp .env.example .env.production
        echo "📝 Please edit .env.production with your API keys"
    fi
    
    if [[ ! -f "config/local_config.yaml" ]]; then
        cp config/trading_config.yaml config/local_config.yaml
        echo "📝 Local configuration created"
    fi
    
    mkdir -p logs data backups
    
    echo "✅ Configuration setup complete"
}

setup_docker() {
    echo "🐳 Setting up Docker..."
    
    if ! command -v docker &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
            open "https://www.docker.com/products/docker-desktop"
        fi
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
    fi
    
    echo "✅ Docker setup complete"
}

setup_kubernetes() {
    echo "☸️  Setting up Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
            rm kubectl
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install kubectl
        fi
    fi
    
    if ! command -v helm &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install helm
        fi
    fi
    
    echo "✅ Kubernetes tools setup complete"
}

setup_monitoring_tools() {
    echo "📊 Setting up monitoring tools..."
    
    mkdir -p monitoring/dashboards
    mkdir -p monitoring/alerts
    
    cat > monitoring/grafana-values.yaml << EOF
persistence:
  enabled: true
  size: 10Gi
adminPassword: quantum_admin
dashboards:
  default:
    quantum-trading:
      gnetId: 15141
      revision: 1
      datasource: Prometheus
EOF

    cat > monitoring/prometheus-values.yaml << EOF
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: standard
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi
grafana:
  enabled: false
alertmanager:
  enabled: true
EOF
    
    echo "✅ Monitoring tools setup complete"
}

generate_ssl_certificates() {
    echo "🔒 Generating SSL certificates..."
    
    mkdir -p ssl
    cd ssl
    
    if [[ ! -f "server.key" ]]; then
        openssl genrsa -out server.key 2048
        openssl req -new -x509 -key server.key -out server.crt -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=quantum-trading"
        
        echo "✅ Self-signed SSL certificates generated"
    else
        echo "✅ SSL certificates already exist"
    fi
    
    cd "$PROJECT_ROOT"
}

setup_git_hooks() {
    echo "🔗 Setting up Git hooks..."
    
    mkdir -p .git/hooks
    
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running pre-commit checks..."
python -m pytest tests/ --quiet || exit 1
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || exit 1
echo "✅ Pre-commit checks passed"
EOF
    
    chmod +x .git/hooks/pre-commit
    
    echo "✅ Git hooks setup complete"
}

run_initial_tests() {
    echo "🧪 Running initial tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    python -m pytest tests/ -v --tb=short
    
    echo "✅ Initial tests completed"
}

create_systemd_service() {
    echo "🔧 Creating systemd service..."
    
    sudo tee /etc/systemd/system/quantum-trading.service > /dev/null << EOF
[Unit]
Description=Quantum Trading System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/venv/bin
ExecStart=$PROJECT_ROOT/venv/bin/python -m quantum_trading_system
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable quantum-trading
    
    echo "✅ Systemd service created"
}

display_completion_message() {
    echo ""
    echo "🎉 Quantum Trading System setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env.production with your API keys"
    echo "2. Review config/local_config.yaml"
    echo "3. Run: ./scripts/deploy.sh production"
    echo "4. Access dashboard at: http://localhost:8050"
    echo ""
    echo "Useful commands:"
    echo "- Start system: sudo systemctl start quantum-trading"
    echo "- View logs: sudo journalctl -u quantum-trading -f"
    echo "- Run tests: source venv/bin/activate && python -m pytest"
    echo "- Docker compose: docker-compose up -d"
    echo ""
}

main() {
    install_system_dependencies
    setup_python_environment
    setup_cuda
    setup_databases
    setup_configuration
    setup_docker
    setup_kubernetes
    setup_monitoring_tools
    generate_ssl_certificates
    setup_git_hooks
    run_initial_tests
    create_systemd_service
    display_completion_message
}

main "$@"