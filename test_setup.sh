#!/bin/bash
set -euo pipefail

echo "Testing Quantum Trading System setup..."

if [ -f .env ]; then
    echo "✅ .env file exists"
    source .env
    echo "✅ Environment loaded"
else
    echo "❌ .env file missing"
    exit 1
fi

if [ -f compliance/data_agreements.json ]; then
    echo "✅ Compliance files exist"
else
    echo "❌ Compliance files missing"
    exit 1
fi

if [ -f simple_server.py ]; then
    echo "✅ Mock server exists"
else
    echo "❌ Mock server missing"
    exit 1
fi

echo ""
echo "🚀 Setup verification complete!"
echo ""
echo "To test:"
echo "1. source .env"
echo "2. python3 simple_server.py"
echo "3. Open http://localhost:8080/health"
echo ""
echo "To run compliance check:"
echo "./compliance_check.sh"
