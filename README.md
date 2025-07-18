# AI-Powered Docker Offload Demo

A complete implementation of GPU-accelerated deep learning inference using NVIDIA Triton, Docker Offload, and agentic architecture patterns.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+ with GPU support
- NVIDIA Container Toolkit
- Docker Compose v2.0+
- NVIDIA GPU (Tesla, RTX, or A100 series)
- 8GB+ GPU memory recommended

### Installation Steps

1. **Clone and Setup**
```bash
git clone <your-repo>
cd ai-docker-offload-demo
chmod +x scripts/setup.sh
./scripts/setup.sh
```

2. **Download Models**
```bash
# Download sample models for testing
./scripts/download-models.sh
```

3. **Start the System**
```bash
# Local GPU deployment
docker-compose up -d

# With Docker Offload (requires offload setup)
docker-compose --profile gpu-offload up -d
```

4. **Verify Installation**
```bash
./scripts/test-system.sh
```

## 🧪 Testing

### Basic Health Check
```bash
# Check all services
curl http://localhost:8080/health

# List available models
curl http://localhost:8080/models
```

### End-to-End Inference Test
```bash
# Text classification
./scripts/test-inference.sh text "Hello, this is a test"

# Image classification  
./scripts/test-inference.sh image ./test-data/sample.jpg

# Audio transcription
./scripts/test-inference.sh audio ./test-data/sample.wav
```

### Performance Benchmarking
```bash
# Load testing with different configurations
./scripts/benchmark.sh --local-gpu
./scripts/benchmark.sh --cloud-offload
./scripts/benchmark.sh --mixed-workload
```

## 📁 Project Structure

```
├── agents/                 # Agentic microservices
│   ├── coordinator/        # Inference coordinator agent
│   ├── preprocessor/       # Data preprocessing agent
│   └── aggregator/         # Results aggregation agent
├── triton-server/          # NVIDIA Triton configuration
├── models/                 # AI model repository
├── scripts/                # Setup and testing scripts
├── docker-compose.yml      # Local deployment
├── docker-offload.yml      # Offload configuration
└── test-data/              # Sample test files
```

## 🔧 Configuration

### GPU Offload Policies

Edit `docker-offload.yml` to customize workload placement:

- **High-performance models** → GPU cluster
- **Preprocessing tasks** → Edge compute
- **Coordination logic** → Local deployment
- **Fallback routing** → Cloud instances

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRITON_URL` | Triton server endpoint | `triton-server:8000` |
| `GPU_MEMORY_LIMIT` | GPU memory allocation | `8Gi` |
| `OFFLOAD_ENABLED` | Enable Docker offload | `false` |
| `CLOUD_ENDPOINT` | Cloud GPU endpoint | `unset` |

## 📊 Monitoring

Access monitoring dashboards:

- **System Health**: http://localhost:8080/health
- **Model Metrics**: http://localhost:8002/metrics
- **Agent Status**: http://localhost:8080/agents

## 🐛 Troubleshooting

### Common Issues

1. **GPU Not Detected**
```bash
# Verify NVIDIA runtime
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

2. **Model Loading Fails**
```bash
# Check model repository
docker logs triton-server
```

3. **Offload Not Working**
```bash
# Verify offload configuration
docker-compose config --profile gpu-offload
```

### Performance Optimization

- Use model ensembles for higher throughput
- Enable dynamic batching in Triton
- Configure memory pool sizes
- Optimize Docker layer caching

## 🌐 Production Deployment

For production environments:

1. Use Kubernetes with GPU operators
2. Implement proper security policies
3. Set up centralized logging
4. Configure auto-scaling policies
5. Monitor costs and performance

## 📚 Additional Resources

- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [Agentic Architecture Patterns](https://docs.example.com/agentic-patterns)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details