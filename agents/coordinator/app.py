# agents/coordinator/app.py - Agentic inference coordinator
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
import numpy as np

import tritonclient.http as httpclient
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
REQUEST_DURATION = Histogram('inference_request_duration_seconds', 'Request duration', ['model'])
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
MODEL_AVAILABILITY = Gauge('model_availability', 'Model availability status', ['model'])

app = FastAPI(title="AI Inference Coordinator Agent", version="1.0.0")

class InferenceRequest(BaseModel):
    model_name: str
    inputs: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None

class AgentStatus(BaseModel):
    status: str
    models_loaded: int
    uptime_seconds: float
    gpu_available: bool

class Agent:
    def __init__(self, triton_url: str):
        self.triton_url = triton_url
        self.client = None
        self.model_metadata = {}
        self.start_time = time.time()
        self.health_status = "initializing"
        
    async def initialize(self):
        """Discover available models and their capabilities"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.client = httpclient.InferenceServerClient(url=self.triton_url)
                
                # Test connection
                if not self.client.is_server_ready():
                    raise Exception("Triton server not ready")
                
                # Load model metadata
                await self._load_model_metadata()
                
                self.health_status = "healthy"
                logger.info("Agent initialized successfully", models_count=len(self.model_metadata))
                return
                
            except Exception as e:
                logger.warning(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    self.health_status = "unhealthy"
                    logger.error("Failed to initialize agent after all retries")
                    raise

    async def _load_model_metadata(self):
        """Load metadata for all available models"""
        try:
            models = self.client.get_model_repository_index()
            self.model_metadata = {}
            
            for model in models:
                if model.state == "READY":
                    try:
                        metadata = self.client.get_model_metadata(model.name)
                        self.model_metadata[model.name] = {
                            'inputs': [{'name': inp.name, 'datatype': inp.datatype, 'shape': inp.shape} 
                                     for inp in metadata.inputs],
                            'outputs': [{'name': out.name, 'datatype': out.datatype, 'shape': out.shape} 
                                      for out in metadata.outputs],
                            'platform': metadata.platform,
                            'max_batch_size': getattr(metadata, 'max_batch_size', 0)
                        }
                        MODEL_AVAILABILITY.labels(model=model.name).set(1)
                        logger.info("Loaded model metadata", model=model.name, platform=metadata.platform)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for model {model.name}: {e}")
                        MODEL_AVAILABILITY.labels(model=model.name).set(0)
                        
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
            raise

    async def route_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Intelligent routing based on model capabilities and load"""
        start_time = time.time()
        
        try:
            if request.model_name not in self.model_metadata:
                REQUEST_COUNT.labels(model=request.model_name, status='error').inc()
                raise HTTPException(404, f"Model {request.model_name} not found or not ready")
                
            # Agentic decision-making for optimal routing
            model_info = self.model_metadata[request.model_name]
            
            # Prepare inputs for Triton
            inputs = []
            for input_spec in model_info['inputs']:
                input_name = input_spec['name']
                if input_name not in request.inputs:
                    raise HTTPException(400, f"Missing required input: {input_name}")
                
                input_data = request.inputs[input_name]
                
                # Handle different input formats
                if isinstance(input_data, dict):
                    data = np.array(input_data.get('data', input_data))
                    shape = input_data.get('shape', data.shape)
                    datatype = input_data.get('datatype', input_spec['datatype'])
                else:
                    data = np.array(input_data)
                    shape = data.shape
                    datatype = input_spec['datatype']
                
                input_tensor = httpclient.InferInput(input_name, shape, datatype)
                input_tensor.set_data_from_numpy(data)
                inputs.append(input_tensor)
            
            # Prepare outputs
            outputs = []
            for output_spec in model_info['outputs']:
                output = httpclient.InferRequestedOutput(output_spec['name'])
                outputs.append(output)
            
            # Execute inference with monitoring
            inference_start = time.time()
            result = self.client.infer(request.model_name, inputs, outputs=outputs)
            inference_time = time.time() - inference_start
            
            # Process results
            response_outputs = {}
            for output_spec in model_info['outputs']:
                output_name = output_spec['name']
                output_data = result.as_numpy(output_name)
                response_outputs[output_name] = output_data.tolist()
            
            execution_time = time.time() - start_time
            
            # Update metrics
            REQUEST_COUNT.labels(model=request.model_name, status='success').inc()
            REQUEST_DURATION.labels(model=request.model_name).observe(execution_time)
            
            response = {
                'model': request.model_name,
                'outputs': response_outputs,
                'metadata': {
                    'execution_time_ms': int(execution_time * 1000),
                    'inference_time_ms': int(inference_time * 1000),
                    'agent_id': 'coordinator-001',
                    'timestamp': time.time()
                }
            }
            
            logger.info("Inference completed", 
                       model=request.model_name, 
                       execution_time=execution_time,
                       inference_time=inference_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            REQUEST_COUNT.labels(model=request.model_name, status='error').inc()
            logger.error("Inference failed", model=request.model_name, error=str(e))
            raise HTTPException(500, f"Inference failed: {str(e)}")

    def get_status(self) -> AgentStatus:
        """Get current agent status"""
        uptime = time.time() - self.start_time
        return AgentStatus(
            status=self.health_status,
            models_loaded=len(self.model_metadata),
            uptime_seconds=uptime,
            gpu_available=self._check_gpu_availability()
        )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            if self.client and self.client.is_server_ready():
                return True
        except:
            pass
        return False

# Global agent instance
coordinator = Agent(triton_url=f"http://{os.getenv('TRITON_URL', 'triton-server:8000')}")

@app.on_event("startup")
async def startup():
    """Initialize the agent on startup"""
    await coordinator.initialize()

@app.post("/infer")
async def infer(request: InferenceRequest):
    """Execute inference request"""
    return await coordinator.route_inference(request)

@app.get("/models")
async def list_models():
    """List available models and their metadata"""
    return {
        "models": list(coordinator.model_metadata.keys()),
        "metadata": coordinator.model_metadata
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = coordinator.get_status()
    return {
        "status": status.status,
        "agent": "inference-coordinator",
        "models_loaded": status.models_loaded,
        "uptime_seconds": status.uptime_seconds,
        "gpu_available": status.gpu_available
    }

@app.get("/status")
async def get_status():
    """Detailed status information"""
    return coordinator.get_status()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/agents")
async def list_agents():
    """List connected agents (for monitoring)"""
    return {
        "agents": [
            {
                "id": "coordinator-001",
                "type": "inference-coordinator",
                "status": coordinator.health_status,
                "models": list(coordinator.model_metadata.keys()),
                "uptime": time.time() - coordinator.start_time
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)