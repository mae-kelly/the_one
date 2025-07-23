#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
NAMESPACE="quantum-trading"

echo "🚀 Starting Quantum Trading System deployment..."
echo "Environment: $ENVIRONMENT"
echo "Namespace: $NAMESPACE"

check_dependencies() {
    echo "📋 Checking dependencies..."
    
    command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl is required but not