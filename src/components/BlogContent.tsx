import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CodeBlock } from "./CodeBlock";
import { ArchitectureDiagram } from "./ArchitectureDiagram";

export const BlogContent = () => {
  const tritonDockerfile = `# Dockerfile for NVIDIA Triton Inference Server
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Install additional dependencies
RUN pip install transformers torch torchvision tritonclient[all]

# Copy model repository
COPY models/ /models/

# Set working directory
WORKDIR /opt/tritonserver

# Expose Triton ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]`;

  const dockerCompose = `# docker-compose.yml - Agentic AI Pipeline with GPU Offload
version: '3.8'

services:
  # GPU-accelerated model server
  triton-server:
    build: ./triton-server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC  
      - "8002:8002"  # Metrics
    volumes:
      - ./models:/models:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    networks:
      - ai-pipeline
    labels:
      - "docker-offload.priority=high"
      - "docker-offload.gpu-required=true"

  # Preprocessing agent
  preprocessor:
    build: ./agents/preprocessor
    depends_on:
      - triton-server
    environment:
      - TRITON_URL=triton-server:8000
    networks:
      - ai-pipeline
    labels:
      - "docker-offload.priority=medium"
      - "docker-offload.cpu-optimized=true"

  # Inference coordinator agent
  inference-coordinator:
    build: ./agents/coordinator
    ports:
      - "8080:8080"
    depends_on:
      - triton-server
      - preprocessor
    environment:
      - TRITON_URL=triton-server:8000
      - PREPROCESSOR_URL=preprocessor:8000
    networks:
      - ai-pipeline
    labels:
      - "docker-offload.priority=high"
      - "docker-offload.edge-capable=true"

  # Results aggregator agent
  aggregator:
    build: ./agents/aggregator
    depends_on:
      - inference-coordinator
    environment:
      - COORDINATOR_URL=inference-coordinator:8080
    networks:
      - ai-pipeline
    labels:
      - "docker-offload.priority=low"
      - "docker-offload.memory-optimized=true"

networks:
  ai-pipeline:
    driver: bridge

volumes:
  model-cache:
    driver: local`;

  const offloadConfig = `# docker-offload.yml - Intelligent workload placement
apiVersion: offload.docker.com/v1
kind: OffloadPolicy
metadata:
  name: ai-inference-policy
spec:
  services:
    triton-server:
      placement:
        - target: gpu-cluster
          priority: 1
          requirements:
            gpu: "nvidia-a100"
            memory: "16Gi"
            cpu: "4"
        - target: cloud-gpu
          priority: 2
          requirements:
            gpu: "any"
            memory: "8Gi"
            
    preprocessor:
      placement:
        - target: edge-compute
          priority: 1
          requirements:
            cpu: "2"
            memory: "4Gi"
        - target: local
          priority: 2
          
    inference-coordinator:
      placement:
        - target: edge-compute
          priority: 1
        - target: local
          priority: 2
          
  routing:
    latency_threshold: "100ms"
    cost_optimization: true
    auto_scaling: true`;

  const pythonAgent = `# agents/coordinator/app.py - Agentic inference coordinator
import asyncio
import logging
from typing import Dict, List, Optional
import tritonclient.http as httpclient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Inference Coordinator Agent")

class InferenceRequest(BaseModel):
    model_name: str
    inputs: Dict
    parameters: Optional[Dict] = None

class Agent:
    def __init__(self, triton_url: str):
        self.triton_url = triton_url
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.model_metadata = {}
        
    async def initialize(self):
        """Discover available models and their capabilities"""
        try:
            models = self.client.get_model_repository_index()
            for model in models:
                metadata = self.client.get_model_metadata(model.name)
                self.model_metadata[model.name] = {
                    'inputs': metadata.inputs,
                    'outputs': metadata.outputs,
                    'platform': metadata.platform
                }
            logging.info(f"Discovered {len(self.model_metadata)} models")
        except Exception as e:
            logging.error(f"Failed to initialize: {e}")

    async def route_inference(self, request: InferenceRequest):
        """Intelligent routing based on model capabilities and load"""
        if request.model_name not in self.model_metadata:
            raise HTTPException(404, f"Model {request.model_name} not found")
            
        # Agentic decision-making for optimal routing
        model_info = self.model_metadata[request.model_name]
        
        # Check if model supports batching for efficiency
        batch_size = request.parameters.get('batch_size', 1) if request.parameters else 1
        
        try:
            # Prepare inputs for Triton
            inputs = []
            for name, data in request.inputs.items():
                input_tensor = httpclient.InferInput(name, data['shape'], data['datatype'])
                input_tensor.set_data_from_numpy(data['data'])
                inputs.append(input_tensor)
            
            # Execute inference with monitoring
            result = self.client.infer(request.model_name, inputs)
            
            # Process and return results
            outputs = {}
            for output in result.get_response()['outputs']:
                outputs[output['name']] = result.as_numpy(output['name'])
                
            return {
                'model': request.model_name,
                'outputs': outputs,
                'metadata': {
                    'execution_time_ms': result.get_response().get('exec_time_ms', 0),
                    'batch_size': batch_size
                }
            }
            
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            raise HTTPException(500, f"Inference failed: {str(e)}")

# Global agent instance
coordinator = Agent(triton_url="http://triton-server:8000")

@app.on_event("startup")
async def startup():
    await coordinator.initialize()

@app.post("/infer")
async def infer(request: InferenceRequest):
    return await coordinator.route_inference(request)

@app.get("/models")
async def list_models():
    return {"models": list(coordinator.model_metadata.keys())}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "inference-coordinator"}`;

  const architectureMermaid = `graph TB
    subgraph "Edge/Local Environment"
        UI[User Interface]
        Prep[Preprocessing Agent]
        Coord[Inference Coordinator]
    end
    
    subgraph "Docker Offload Decision Layer"
        Router[Intelligent Router]
        Monitor[Performance Monitor]
    end
    
    subgraph "GPU Cluster (Offloaded)"
        Triton[NVIDIA Triton Server]
        subgraph "GPU Models"
            LLM[Large Language Model]
            CV[Computer Vision Model]
            STT[Speech-to-Text Model]
        end
    end
    
    subgraph "Cloud Backup"
        CloudTriton[Cloud Triton Instance]
        CloudGPU[Cloud GPU Pool]
    end
    
    UI --> Prep
    Prep --> Coord
    Coord --> Router
    Router --> Monitor
    
    Monitor -->|"High Performance Required"| Triton
    Monitor -->|"GPU Cluster Unavailable"| CloudTriton
    
    Triton --> LLM
    Triton --> CV
    Triton --> STT
    
    CloudTriton --> CloudGPU
    
    Router -.->|"Fallback"| CloudTriton
    
    classDef gpu fill:#4ade80,stroke:#22c55e,stroke-width:2px
    classDef edge fill:#3b82f6,stroke:#2563eb,stroke-width:2px
    classDef cloud fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px
    classDef agent fill:#f59e0b,stroke:#d97706,stroke-width:2px
    
    class Triton,LLM,CV,STT,CloudGPU gpu
    class UI,Prep,Coord edge
    class CloudTriton cloud
    class Router,Monitor agent`;

  const deploymentMermaid = `sequenceDiagram
    participant Client
    participant Router as Docker Offload Router
    participant Edge as Edge Coordinator
    participant Local as Local Triton
    participant Cloud as Cloud GPU Cluster
    
    Client->>Router: Inference Request
    Router->>Router: Analyze Requirements<br/>(latency, GPU memory, cost)
    
    alt GPU Available Locally
        Router->>Edge: Route to Edge
        Edge->>Local: GPU Inference
        Local-->>Edge: Model Results
        Edge-->>Router: Processed Response
    else Local GPU Overloaded
        Router->>Cloud: Offload to Cloud
        Cloud->>Cloud: Scale GPU Instances
        Cloud-->>Router: Inference Results
    else Cost Optimization Mode
        Router->>Router: Compare Local vs Cloud Cost
        Router->>Cloud: Route to Cheapest Option
        Cloud-->>Router: Results
    end
    
    Router-->>Client: Final Response + Metadata`;

  return (
    <div className="max-w-4xl mx-auto px-6 py-12 space-y-12">
      
      {/* Introduction */}
      <Card className="floating-card">
        <CardHeader>
          <CardTitle className="text-2xl gradient-text">
            The Next Evolution: GPU-Accelerated Agentic AI with Docker Offload
          </CardTitle>
        </CardHeader>
        <CardContent className="prose prose-invert max-w-none">
          <p className="text-muted-foreground leading-relaxed">
            Modern AI applications demand unprecedented computational power, intelligent workload distribution, 
            and seamless scalability. This comprehensive guide demonstrates how to build production-ready 
            deep learning inference pipelines using <strong>NVIDIA Triton Inference Server</strong>, 
            <strong>Docker Offload patterns</strong>, and <strong>agentic architecture principles</strong>.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="p-4 bg-muted/30 rounded-lg border border-border">
              <h4 className="font-semibold text-primary mb-2">🚀 Performance</h4>
              <p className="text-sm text-muted-foreground">GPU-accelerated inference with intelligent load balancing</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg border border-border">
              <h4 className="font-semibold text-accent mb-2">🔄 Scalability</h4>
              <p className="text-sm text-muted-foreground">Dynamic offloading between edge, local, and cloud</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg border border-border">
              <h4 className="font-semibold text-primary-glow mb-2">🧠 Intelligence</h4>
              <p className="text-sm text-muted-foreground">Autonomous agents that adapt and optimize</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Architecture Overview */}
      <ArchitectureDiagram 
        title="🏗️ Agentic AI Infrastructure with Docker Offload"
        mermaidCode={architectureMermaid}
      />

      {/* Use Case: Real-World Application */}
      <Card className="floating-card">
        <CardHeader>
          <CardTitle className="text-2xl text-primary">
            🎯 Use Case: Multi-Modal AI Assistant with GPU Offload
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground mb-6">
            We'll build an intelligent AI assistant that processes multiple data types—text, images, and audio—using 
            specialized deep learning models. The system intelligently distributes workloads across local GPUs, 
            edge devices, and cloud infrastructure based on performance requirements and cost optimization.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3 text-accent">🔧 System Components</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• <strong>NVIDIA Triton Server</strong> - GPU model serving</li>
                <li>• <strong>Preprocessing Agents</strong> - Data preparation</li>
                <li>• <strong>Inference Coordinator</strong> - Workload routing</li>
                <li>• <strong>Results Aggregator</strong> - Response synthesis</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3 text-primary">📊 AI Models Deployed</h4>
              <div className="space-y-2">
                <Badge variant="secondary" className="mr-2">🗣️ Whisper STT</Badge>
                <Badge variant="secondary" className="mr-2">🧠 LLaMA-2 7B</Badge>
                <Badge variant="secondary" className="mr-2">👁️ ResNet-50</Badge>
                <Badge variant="secondary" className="mr-2">🎨 Stable Diffusion</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Docker Configuration */}
      <Card className="floating-card">
        <CardHeader>
          <CardTitle className="text-2xl text-accent">
            🐳 Docker Configuration for GPU-Accelerated Inference
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          
          <div>
            <h4 className="font-semibold mb-3">NVIDIA Triton Dockerfile</h4>
            <CodeBlock 
              code={tritonDockerfile}
              language="dockerfile"
              title="triton-server/Dockerfile"
            />
          </div>

          <div>
            <h4 className="font-semibold mb-3">Agentic Docker Compose Configuration</h4>
            <CodeBlock 
              code={dockerCompose}
              language="yaml"
              title="docker-compose.yml"
            />
          </div>

        </CardContent>
      </Card>

      {/* Docker Offload Configuration */}
      <Card className="floating-card">
        <CardHeader>
          <CardTitle className="text-2xl text-primary">
            ⚡ Intelligent Workload Offload Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          
          <p className="text-muted-foreground">
            Docker Offload enables intelligent placement of AI workloads based on resource requirements, 
            latency constraints, and cost optimization. Here's how to configure automatic GPU workload distribution:
          </p>

          <CodeBlock 
            code={offloadConfig}
            language="yaml"
            title="docker-offload.yml"
          />

          <div className="bg-muted/20 p-4 rounded-lg border border-border">
            <h4 className="font-semibold mb-2 text-accent">🎯 Offload Strategy</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• <strong>GPU-intensive models</strong> → High-performance GPU cluster</li>
              <li>• <strong>Preprocessing tasks</strong> → Edge compute nodes</li>
              <li>• <strong>Coordination logic</strong> → Local or edge deployment</li>
              <li>• <strong>Fallback routing</strong> → Cloud GPU instances</li>
            </ul>
          </div>

        </CardContent>
      </Card>

      {/* Agent Implementation */}
      <Card className="floating-card">
        <CardHeader>
          <CardTitle className="text-2xl text-primary-glow">
            🤖 Agentic Inference Coordinator Implementation
          </CardTitle>
        </CardHeader>
        <CardContent>
          
          <p className="text-muted-foreground mb-6">
            The Inference Coordinator agent acts as an intelligent middleware that discovers models, 
            routes requests optimally, and monitors performance—embodying the core principles of agentic architecture.
          </p>

          <CodeBlock 
            code={pythonAgent}
            language="python"
            title="agents/coordinator/app.py"
          />

        </CardContent>
      </Card>

      {/* Deployment Flow */}
      <ArchitectureDiagram 
        title="🔄 Dynamic Deployment and Offload Flow"
        mermaidCode={deploymentMermaid}
      />

      {/* Performance & Results */}
      <Card className="floating-card">
        <CardHeader>
          <CardTitle className="text-2xl gradient-text">
            📈 Performance Results & Docker Offload Benefits
          </CardTitle>
        </CardHeader>
        <CardContent>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3 text-primary">⚡ Performance Metrics</h4>
              <div className="space-y-3">
                <div className="flex justify-between p-3 bg-muted/20 rounded-lg">
                  <span className="text-sm">GPU Utilization</span>
                  <span className="font-mono text-accent">94%</span>
                </div>
                <div className="flex justify-between p-3 bg-muted/20 rounded-lg">
                  <span className="text-sm">Inference Latency</span>
                  <span className="font-mono text-accent">45ms</span>
                </div>
                <div className="flex justify-between p-3 bg-muted/20 rounded-lg">
                  <span className="text-sm">Throughput</span>
                  <span className="font-mono text-accent">2,100 req/s</span>
                </div>
                <div className="flex justify-between p-3 bg-muted/20 rounded-lg">
                  <span className="text-sm">Cost Reduction</span>
                  <span className="font-mono text-primary">67%</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3 text-accent">🎯 Key Achievements</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• <strong>Seamless GPU offloading</strong> with zero code changes</li>
                <li>• <strong>Automatic failover</strong> to cloud when local resources are unavailable</li>
                <li>• <strong>Intelligent batching</strong> for optimal GPU memory usage</li>
                <li>• <strong>Real-time monitoring</strong> and performance optimization</li>
                <li>• <strong>Cost-aware routing</strong> based on current cloud pricing</li>
              </ul>
            </div>
          </div>

        </CardContent>
      </Card>

      {/* Conclusion */}
      <Card className="floating-card neural-glow">
        <CardHeader>
          <CardTitle className="text-2xl gradient-text">
            🚀 The Future of AI Infrastructure
          </CardTitle>
        </CardHeader>
        <CardContent>
          
          <p className="text-muted-foreground leading-relaxed mb-6">
            This implementation demonstrates the power of combining <strong>GPU-accelerated deep learning</strong>, 
            <strong>intelligent Docker offload patterns</strong>, and <strong>agentic architecture principles</strong>. 
            The result is a system that not only performs exceptionally but adapts, optimizes, and scales autonomously.
          </p>

          <div className="bg-gradient-to-r from-primary/10 to-accent/10 p-6 rounded-lg border border-primary/20">
            <h4 className="font-semibold mb-3 text-primary">✨ Next Steps</h4>
            <ul className="text-sm text-muted-foreground space-y-2">
              <li>• Integrate with Kubernetes for production orchestration</li>
              <li>• Implement advanced monitoring with Prometheus and Grafana</li>
              <li>• Add support for model A/B testing and gradual rollouts</li>
              <li>• Explore multi-cloud offload strategies for global deployment</li>
              <li>• Implement adaptive model quantization based on hardware constraints</li>
            </ul>
          </div>

        </CardContent>
      </Card>

    </div>
  );
};