#!/bin/bash

# Inference testing script
set -e

COORDINATOR_URL="http://localhost:8080"
PREPROCESSOR_URL="http://localhost:8080"

# Function to test text inference
test_text() {
    local text="$1"
    echo "🔤 Testing text inference: '$text'"
    
    local payload=$(cat << EOF
{
    "data_type": "text",
    "data": "$text",
    "target_model": "text_classifier"
}
EOF
)
    
    echo "Preprocessing text..."
    if preprocess_response=$(curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$PREPROCESSOR_URL/preprocess"); then
        
        echo "✅ Text preprocessing successful"
        echo "   Response: $(echo "$preprocess_response" | jq -c '.metadata // {}' 2>/dev/null)"
        
        # Extract preprocessed data for inference
        local inference_payload=$(echo "$preprocess_response" | jq '{
            model_name: "text_classifier",
            inputs: .preprocessed_data
        }')
        
        echo "Running inference..."
        if inference_response=$(curl -s -f -X POST \
            -H "Content-Type: application/json" \
            -d "$inference_payload" \
            "$COORDINATOR_URL/infer"); then
            
            echo "✅ Text inference successful"
            echo "   Result: $(echo "$inference_response" | jq -c '.metadata // {}' 2>/dev/null)"
        else
            echo "❌ Text inference failed"
        fi
    else
        echo "❌ Text preprocessing failed"
    fi
}

# Function to test image inference
test_image() {
    local image_path="$1"
    
    if [ ! -f "$image_path" ]; then
        echo "❌ Image file not found: $image_path"
        return 1
    fi
    
    echo "🖼️  Testing image inference: $image_path"
    
    # Encode image to base64
    local image_b64=$(base64 -w 0 "$image_path")
    
    local payload=$(cat << EOF
{
    "data_type": "image",
    "data": "$image_b64",
    "target_model": "resnet50"
}
EOF
)
    
    echo "Preprocessing image..."
    if preprocess_response=$(curl -s -f -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$PREPROCESSOR_URL/preprocess"); then
        
        echo "✅ Image preprocessing successful"
        echo "   Response: $(echo "$preprocess_response" | jq -c '.metadata // {}' 2>/dev/null)"
        
        # Extract preprocessed data for inference
        local inference_payload=$(echo "$preprocess_response" | jq '{
            model_name: "resnet50",
            inputs: .preprocessed_data
        }')
        
        echo "Running inference..."
        if inference_response=$(curl -s -f -X POST \
            -H "Content-Type: application/json" \
            -d "$inference_payload" \
            "$COORDINATOR_URL/infer"); then
            
            echo "✅ Image inference successful"
            echo "   Result: $(echo "$inference_response" | jq -c '.metadata // {}' 2>/dev/null)"
        else
            echo "❌ Image inference failed"
        fi
    else
        echo "❌ Image preprocessing failed"
    fi
}

# Function to test file upload
test_file_upload() {
    local file_path="$1"
    local model="$2"
    
    if [ ! -f "$file_path" ]; then
        echo "❌ File not found: $file_path"
        return 1
    fi
    
    echo "📎 Testing file upload: $file_path"
    
    if response=$(curl -s -f -X POST \
        -F "file=@$file_path" \
        -F "target_model=$model" \
        "$PREPROCESSOR_URL/preprocess/file"); then
        
        echo "✅ File upload and preprocessing successful"
        echo "   Response: $(echo "$response" | jq -c '.metadata // {}' 2>/dev/null)"
    else
        echo "❌ File upload failed"
    fi
}

# Main execution
echo "🧪 AI Inference Testing Suite"
echo "=============================="

# Check if services are running
if ! curl -s -f "$COORDINATOR_URL/health" > /dev/null; then
    echo "❌ Coordinator service not available at $COORDINATOR_URL"
    echo "   Please ensure services are running: docker-compose up -d"
    exit 1
fi

# Test based on arguments
case "${1:-all}" in
    "text")
        text="${2:-Hello, this is a test message for AI inference}"
        test_text "$text"
        ;;
    "image")
        image_path="${2:-./test-data/sample.jpg}"
        test_image "$image_path"
        ;;
    "file")
        file_path="${2:-./test-data/sample.jpg}"
        model="${3:-resnet50}"
        test_file_upload "$file_path" "$model"
        ;;
    "all"|*)
        echo "Running all tests..."
        echo ""
        
        # Test text
        test_text "This is a sample text for AI classification and analysis"
        echo ""
        
        # Test image if available
        if [ -f "./test-data/sample.jpg" ]; then
            test_image "./test-data/sample.jpg"
        else
            echo "⚠️  No sample image found, skipping image test"
        fi
        echo ""
        
        # Test file upload
        if [ -f "./test-data/sample.txt" ]; then
            test_file_upload "./test-data/sample.txt" "text_classifier"
        else
            echo "⚠️  No sample text file found, skipping file upload test"
        fi
        ;;
esac

echo ""
echo "🎉 Testing complete!"
echo ""
echo "💡 Usage examples:"
echo "   ./scripts/test-inference.sh text 'Your custom text here'"
echo "   ./scripts/test-inference.sh image /path/to/image.jpg"
echo "   ./scripts/test-inference.sh file /path/to/file.txt text_classifier"