# agents/preprocessor/app.py - Data preprocessing agent
import io
import base64
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
import librosa
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

app = FastAPI(title="Preprocessing Agent", version="1.0.0")

class PreprocessRequest(BaseModel):
    data_type: str  # 'text', 'image', 'audio'
    data: str  # base64 encoded or text
    target_model: str
    parameters: Optional[Dict[str, Any]] = None

class PreprocessResponse(BaseModel):
    preprocessed_data: Dict[str, Any]
    metadata: Dict[str, Any]

class PreprocessorAgent:
    def __init__(self):
        self.supported_models = {
            'text_classifier': self._preprocess_text,
            'image_classifier': self._preprocess_image,
            'speech_to_text': self._preprocess_audio,
            'llama2': self._preprocess_text,
            'whisper': self._preprocess_audio,
            'resnet50': self._preprocess_image
        }
    
    async def preprocess(self, request: PreprocessRequest) -> PreprocessResponse:
        """Route preprocessing based on target model"""
        try:
            if request.target_model not in self.supported_models:
                raise HTTPException(400, f"Model {request.target_model} not supported")
            
            preprocess_func = self.supported_models[request.target_model]
            processed_data = await preprocess_func(request)
            
            return PreprocessResponse(
                preprocessed_data=processed_data,
                metadata={
                    'agent_id': 'preprocessor-001',
                    'target_model': request.target_model,
                    'data_type': request.data_type
                }
            )
            
        except Exception as e:
            logger.error("Preprocessing failed", error=str(e), model=request.target_model)
            raise HTTPException(500, f"Preprocessing failed: {str(e)}")
    
    async def _preprocess_text(self, request: PreprocessRequest) -> Dict[str, Any]:
        """Preprocess text data"""
        text = request.data
        
        # Basic text preprocessing
        text = text.strip()
        
        # For LLaMA-2, format as tokens (simplified)
        if 'llama' in request.target_model.lower():
            # Simple tokenization (in production, use proper tokenizer)
            tokens = text.split()
            token_ids = [hash(token) % 50000 for token in tokens]  # Mock tokenization
            
            return {
                'INPUT_IDS': {
                    'data': [token_ids],
                    'shape': [1, len(token_ids)],
                    'datatype': 'INT32'
                },
                'ATTENTION_MASK': {
                    'data': [[1] * len(token_ids)],
                    'shape': [1, len(token_ids)],
                    'datatype': 'INT32'
                }
            }
        else:
            # For simple text classification
            # Convert text to embedding (mock implementation)
            embedding = [ord(c) / 255.0 for c in text[:512]]  # Mock embedding
            embedding.extend([0.0] * (512 - len(embedding)))  # Pad to 512
            
            return {
                'INPUT': {
                    'data': [embedding],
                    'shape': [1, 512],
                    'datatype': 'FP32'
                }
            }
    
    async def _preprocess_image(self, request: PreprocessRequest) -> Dict[str, Any]:
        """Preprocess image data"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(request.data)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model requirements (ResNet-50 expects 224x224)
            if 'resnet' in request.target_model.lower():
                image = image.resize((224, 224))
                
                # Convert to numpy array and normalize
                img_array = np.array(image).astype(np.float32)
                img_array = img_array / 255.0  # Normalize to [0, 1]
                
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                
                # Add batch dimension and transpose to CHW format
                img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
                img_array = np.expand_dims(img_array, axis=0)   # Add batch dim
                
                return {
                    'INPUT': {
                        'data': img_array.tolist(),
                        'shape': list(img_array.shape),
                        'datatype': 'FP32'
                    }
                }
            else:
                # Generic image preprocessing
                image = image.resize((256, 256))
                img_array = np.array(image).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                return {
                    'INPUT': {
                        'data': img_array.tolist(),
                        'shape': list(img_array.shape),
                        'datatype': 'FP32'
                    }
                }
                
        except Exception as e:
            raise HTTPException(400, f"Invalid image data: {str(e)}")
    
    async def _preprocess_audio(self, request: PreprocessRequest) -> Dict[str, Any]:
        """Preprocess audio data"""
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(request.data)
            
            # Load audio using librosa
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
            
            if 'whisper' in request.target_model.lower():
                # Whisper expects specific format
                # Pad or trim to 30 seconds at 16kHz
                target_length = 30 * 16000
                if len(audio_array) > target_length:
                    audio_array = audio_array[:target_length]
                else:
                    audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
                
                # Add batch dimension
                audio_array = np.expand_dims(audio_array, axis=0)
                
                return {
                    'AUDIO': {
                        'data': audio_array.tolist(),
                        'shape': list(audio_array.shape),
                        'datatype': 'FP32'
                    }
                }
            else:
                # Generic audio preprocessing - extract MFCC features
                mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
                mfccs = np.expand_dims(mfccs.T, axis=0)  # Add batch dimension
                
                return {
                    'INPUT': {
                        'data': mfccs.tolist(),
                        'shape': list(mfccs.shape),
                        'datatype': 'FP32'
                    }
                }
                
        except Exception as e:
            raise HTTPException(400, f"Invalid audio data: {str(e)}")

# Global agent instance
preprocessor = PreprocessorAgent()

@app.post("/preprocess")
async def preprocess_data(request: PreprocessRequest):
    """Preprocess data for inference"""
    return await preprocessor.preprocess(request)

@app.post("/preprocess/file")
async def preprocess_file(file: UploadFile = File(...), target_model: str = "resnet50"):
    """Preprocess uploaded file"""
    try:
        content = await file.read()
        data_b64 = base64.b64encode(content).decode('utf-8')
        
        # Determine data type from file extension
        data_type = "image"
        if file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            data_type = "audio"
        elif file.filename.lower().endswith('.txt'):
            data_type = "text"
            data_b64 = content.decode('utf-8')  # Text doesn't need base64
        
        request = PreprocessRequest(
            data_type=data_type,
            data=data_b64,
            target_model=target_model
        )
        
        return await preprocessor.preprocess(request)
        
    except Exception as e:
        raise HTTPException(500, f"File preprocessing failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "preprocessor",
        "supported_models": list(preprocessor.supported_models.keys())
    }

@app.get("/models")
async def supported_models():
    """List supported models for preprocessing"""
    return {
        "supported_models": list(preprocessor.supported_models.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)