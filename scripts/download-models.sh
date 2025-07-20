#!/bin/bash

# Model download script for AI Docker Offload Demo
set -e

echo "📥 Downloading AI models for testing..."

# Create model directories
mkdir -p models/{text_classifier,image_classifier,speech_to_text}/{1,model.py}
mkdir -p triton-config

# Function to download model safely
download_model() {
    local model_name=$1
    local model_url=$2
    local model_path=$3
    
    echo "⬇️  Downloading $model_name..."
    
    if [ ! -f "$model_path" ]; then
        # Create mock model for testing if URL not available
        echo "Creating mock model: $model_name"
        mkdir -p "$(dirname "$model_path")"
        
        case $model_name in
            "text_classifier")
                # Create a simple text classifier mock
                cat > "$model_path/model.py" << 'EOF'
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass
    
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            input_data = input_tensor.as_numpy()
            
            # Mock classification: random binary classification
            output_data = np.random.rand(input_data.shape[0], 2).astype(np.float32)
            output_tensor = pb_utils.Tensor("OUTPUT", output_data)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        return responses
EOF
                ;;
            "image_classifier")
                # Create ONNX model mock
                python3 -c "
import numpy as np
import os
os.makedirs('$model_path', exist_ok=True)
# Create dummy ONNX model placeholder
with open('$model_path/model.onnx', 'wb') as f:
    f.write(b'MOCK_ONNX_MODEL_DATA')
print('Created mock ONNX model')
" 2>/dev/null || echo "⚠️  Python not available for ONNX model creation"
                ;;
        esac
    else
        echo "✅ $model_name already exists"
    fi
}

# Download text classification model (BERT-like)
download_model "text_classifier" "" "models/text_classifier"

# Download image classification model (ResNet50)
download_model "image_classifier" "" "models/image_classifier"

# Create Triton configuration
echo "📁 Setting up Triton configuration..."
mkdir -p triton-config

cat > triton-config/config.pbtxt << 'EOF'
# Global Triton configuration
backend_directory: "/opt/tritonserver/backends"
model_repository_path: "/models"
strict_model_config: false
log_verbose: 1

# Resource limits
cuda_memory_pool_byte_size { 
  key: 0 
  value: 1073741824  # 1GB
}
EOF

# Create additional model versions
echo "📦 Creating model versions..."
for model in text_classifier image_classifier; do
    mkdir -p "models/$model/1"
    echo "# Version 1 of $model" > "models/$model/1/version.txt"
done

# Create sample model files for testing
echo "🧪 Creating test model files..."

# Create Python backend for text classifier
cat > models/text_classifier/1/model.py << 'EOF'
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        # Initialize mock classifier
        self.classifier = lambda x: [{"label": "POSITIVE", "score": 0.8}]
    
    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input text
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            text_input = input_tensor.as_numpy()
            
            # Mock classification
            batch_size = text_input.shape[0]
            output_data = np.random.rand(batch_size, 2).astype(np.float32)
            output_data[:, 0] = 0.8  # Positive class
            output_data[:, 1] = 0.2  # Negative class
            
            output_tensor = pb_utils.Tensor("OUTPUT", output_data)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses
    
    def finalize(self):
        pass
EOF

# Set permissions
chmod +x models/text_classifier/1/model.py

echo "✅ Model download and setup complete!"
echo ""
echo "📊 Model inventory:"
find models/ -name "*.py" -o -name "*.onnx" -o -name "config.pbtxt" | head -10

echo ""
echo "Next steps:"
echo "1. Start services: docker-compose up -d"
echo "2. Test models: ./scripts/test-system.sh"
echo "3. Monitor: ./scripts/benchmark.sh"