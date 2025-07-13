#!/bin/bash

# Medical RAG Docker Test Script
# Script to test application endpoints

set -e

echo "🧪 Medical RAG Application Testing"
echo "=================================="

BASE_URL="http://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test endpoint
test_endpoint() {
    local endpoint=$1
    local description=$2
    local method=${3:-GET}
    local data=${4:-}
    
    echo -n "Testing $description... "
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" 2>/dev/null || echo "000")
    else
        response=$(curl -s -w "%{http_code}" "$BASE_URL$endpoint" 2>/dev/null || echo "000")
    fi
    
    # Extract status code (last 3 characters)
    status_code="${response: -3}"
    
    if [ "$status_code" = "200" ] || [ "$status_code" = "201" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
    else
        echo -e "${RED}❌ FAIL (HTTP $status_code)${NC}"
    fi
}

# Check if application is running
echo "🔍 Checking if application is running..."
if ! curl -f "$BASE_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Application is not running or not accessible${NC}"
    echo "   Start it first with: ./scripts/docker-setup.sh start"
    exit 1
fi

echo -e "${GREEN}✅ Application is running${NC}"
echo ""

# Test basic endpoints
echo "📋 Testing basic endpoints:"
test_endpoint "/health" "Health Check"
test_endpoint "/" "Root endpoint"
test_endpoint "/docs" "API Documentation"
echo ""

# Test knowledge graph endpoints
echo "🧠 Testing knowledge graph endpoints:"
test_endpoint "/knowledge-graph/stats" "Knowledge Graph Stats"
test_endpoint "/knowledge-graph/nodes" "Knowledge Graph Nodes"
test_endpoint "/knowledge-graph/edges" "Knowledge Graph Edges"
echo ""

# Test RAG endpoints
echo "🔍 Testing RAG endpoints:"
test_endpoint "/rag-ready/stats" "RAG Ready Stats"

# Test a simple query
echo "💬 Testing query endpoint..."
test_endpoint "/query-rag-ready" "Query RAG Ready" "POST" '{"query": "What is diabetes?", "max_results": 3}'
echo ""

# Test entity extraction
echo "🏷️  Testing entity extraction:"
test_endpoint "/extract-entities" "Entity Extraction" "POST" '{"text": "Patient has diabetes and takes metformin for treatment."}'
echo ""

echo "🎉 Testing completed!"
echo ""
echo "📊 Summary:"
echo "   - All endpoints tested"
echo "   - Check the output above for any failures"
echo ""
echo "🌐 Access your application at:"
echo "   - Web Interface: $BASE_URL"
echo "   - API Documentation: $BASE_URL/docs"
echo "   - Health Check: $BASE_URL/health" 