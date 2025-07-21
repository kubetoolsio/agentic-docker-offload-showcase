#!/bin/bash

# System testing script for AI Docker Offload Demo
set -e

COORDINATOR_URL="http://localhost:8080"
PREPROCESSOR_URL="http://localhost:8080"  # Through coordinator

echo "🧪 Testing AI Docker Offload System..."

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $((max_attempts * 10)) seconds"
    return 1
}

# Function to test endpoint
test_endpoint() {
    local url=$1
    local description=$2
    
    echo "🔍 Testing: $description"
    
    if response=$(curl -s -f "$url"); then
        echo "✅ $description - OK"
        echo "   Response: $(echo "$response" | jq -c . 2>/dev/null || echo "$response")"
        return 0
    else
        echo "❌ $description - FAILED"
        return 1
    fi
}

# Function to test inference
test_inference() {
    local model=$1
    local input_data=$2
    local description=$3
    
    echo "🔍 Testing inference: $description"
    
    local payload=$(cat << EOF
{
    "model_name": "$model",
    "inputs": $input_data
}
EOF
)
    
    if response=$(curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$COORDINATOR_URL/infer"); then
        echo "✅ $description - OK"
        echo "   Model: $model"
        echo "   Response: $(echo "$response" | jq -c '.metadata // {}' 2>/dev/null || echo "No metadata")"
        return 0
    else
        echo "❌ $description - FAILED"
        echo "   Payload: $payload"
        return 1
    fi
}

# Start testing
echo "Starting system tests at $(date)"
echo "================================="

# Check if Docker Compose is running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Docker Compose services are not running"
    echo "   Please run: docker-compose up -d"
    exit 1
fi

# Wait for services to be ready
wait_for_service "$COORDINATOR_URL" "Inference Coordinator" || exit 1

# Test basic endpoints
echo ""
echo "🔍 Testing basic endpoints..."
test_endpoint "$COORDINATOR_URL/health" "Health check"
test_endpoint "$COORDINATOR_URL/models" "Model listing"
test_endpoint "$COORDINATOR_URL/agents" "Agent listing"

# Test mock inference (since we don't have real models)
echo ""
echo "🔍 Testing mock inference..."

# Text classification test
text_input='{
    "INPUT": {
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        "shape": [1, 5],
        "datatype": "FP32"
    }
}'

# Note: These tests will likely fail without real models, but they test the pipeline
test_inference "text_classifier" "$text_input" "Text classification"

# Image classification test
image_input='{
    "INPUT": {
        "data": [[[[[0.5, 0.6, 0.7]]]]],
        "shape": [1, 3, 1, 1],
        "datatype": "FP32"
    }
}'

test_inference "resnet50" "$image_input" "Image classification"

# Test metrics endpoint
echo ""
echo "🔍 Testing monitoring..."
test_endpoint "$COORDINATOR_URL/metrics" "Prometheus metrics"

# Check Docker containers
echo ""
echo "📊 Container status:"
docker-compose ps

# Check GPU usage if available
echo ""
echo "🖥️  GPU status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "   NVIDIA GPU tools not available"
fi

# Service logs summary
echo ""
echo "📋 Recent service logs:"
echo "--- Coordinator ---"
docker-compose logs --tail=5 inference-coordinator

echo "--- Triton Server ---"
docker-compose logs --tail=5 triton-server

echo ""
echo "================================="
echo "🎉 System test complete!"
echo ""
echo "📊 Access points:"
echo "   • API Endpoint: $COORDINATOR_URL"
echo "   • Health Check: $COORDINATOR_URL/health"
echo "   • Metrics: $COORDINATOR_URL/metrics"
echo "   • Triton Metrics: http://localhost:8002/metrics"
echo ""
echo "📚 Next steps:"
echo "   • Run load tests: ./scripts/benchmark.sh"
echo "   • Test with real data: ./scripts/test-inference.sh"
echo "   • Monitor performance: http://localhost:3000"