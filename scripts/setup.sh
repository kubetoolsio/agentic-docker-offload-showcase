#!/bin/bash

# Setup script for AI Docker Offload Demo
set -e

echo "🚀 Setting up AI Docker Offload Demo..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Engine 20.10+"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose v2.0+"
    exit 1
fi

# Check NVIDIA Docker support
if ! docker run --gpus all --rm nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "⚠️  GPU support not available. Some features will run in CPU mode."
    echo "   To enable GPU support, install NVIDIA Container Toolkit:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
else
    echo "✅ GPU support detected"
fi

# Create directories
echo "📁 Creating project structure..."
mkdir -p models/{text_classifier,image_classifier,speech_to_text}
mkdir -p test-data
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p logs

# Download sample models (mock implementation)
echo "📥 Setting up model repository..."
cat > models/text_classifier/config.pbtxt << 'EOF'
name: "text_classifier"
platform: "python"
max_batch_size: 8
input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]
EOF

cat > models/image_classifier/config.pbtxt << 'EOF'
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
EOF

# Create monitoring configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['triton-server:8002']
  
  - job_name: 'coordinator'
    static_configs:
      - targets: ['inference-coordinator:8080']
    metrics_path: '/metrics'
EOF

# Create test data
echo "🧪 Creating test data..."
echo "This is a sample text for testing the AI inference pipeline" > test-data/sample.txt

# Download a sample image (create a simple test image)
python3 -c "
import numpy as np
from PIL import Image
import os

# Create a simple test image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(img)
image.save('test-data/sample.jpg')
print('Created sample test image')
" 2>/dev/null || echo "⚠️  Could not create sample image (PIL not available)"

# Create environment file
cat > .env << 'EOF'
# AI Docker Offload Configuration
TRITON_URL=triton-server:8000
PREPROCESSOR_URL=preprocessor:8000
COORDINATOR_URL=inference-coordinator:8080
AGGREGATOR_URL=aggregator:8000

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
GPU_MEMORY_LIMIT=8Gi

# Offload Configuration
OFFLOAD_ENABLED=false
CLOUD_ENDPOINT=

# Logging
LOG_LEVEL=INFO
EOF

# Set permissions
chmod +x scripts/*.sh 2>/dev/null || true

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start the system: docker-compose up -d"
echo "2. Wait for services to initialize (2-3 minutes)"
echo "3. Test the system: ./scripts/test-system.sh"
echo "4. View monitoring: http://localhost:3000 (admin/admin)"
echo ""
echo "For GPU-accelerated inference, ensure NVIDIA Container Toolkit is installed."