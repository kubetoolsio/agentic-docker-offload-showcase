# agents/aggregator/app.py - Results aggregation agent
import time
import asyncio
from typing import Dict, List, Any, Optional
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog
import httpx

logger = structlog.get_logger()

app = FastAPI(title="Aggregator Agent", version="1.0.0")

class AggregationRequest(BaseModel):
    results: List[Dict[str, Any]]
    aggregation_type: str = "default"  # 'default', 'ensemble', 'weighted'
    weights: Optional[List[float]] = None

class AggregatedResponse(BaseModel):
    aggregated_results: Dict[str, Any]
    metadata: Dict[str, Any]
    individual_results: List[Dict[str, Any]]

class AggregatorAgent:
    def __init__(self, coordinator_url: str):
        self.coordinator_url = coordinator_url
        self.aggregation_strategies = {
            'default': self._default_aggregation,
            'ensemble': self._ensemble_aggregation,
            'weighted': self._weighted_aggregation,
            'confidence': self._confidence_based_aggregation
        }
    
    async def aggregate_results(self, request: AggregationRequest) -> AggregatedResponse:
        """Aggregate multiple inference results"""
        try:
            if not request.results:
                raise HTTPException(400, "No results to aggregate")
            
            strategy = request.aggregation_type
            if strategy not in self.aggregation_strategies:
                strategy = 'default'
            
            aggregation_func = self.aggregation_strategies[strategy]
            aggregated = await aggregation_func(request.results, request.weights)
            
            return AggregatedResponse(
                aggregated_results=aggregated,
                metadata={
                    'agent_id': 'aggregator-001',
                    'aggregation_type': strategy,
                    'num_results': len(request.results),
                    'timestamp': time.time()
                },
                individual_results=request.results
            )
            
        except Exception as e:
            logger.error("Aggregation failed", error=str(e))
            raise HTTPException(500, f"Aggregation failed: {str(e)}")
    
    async def _default_aggregation(self, results: List[Dict], weights: Optional[List[float]]) -> Dict[str, Any]:
        """Simple majority vote or average aggregation"""
        if len(results) == 1:
            return results[0]['outputs']
        
        # For multiple results, take the first one (can be enhanced)
        return results[0]['outputs']
    
    async def _ensemble_aggregation(self, results: List[Dict], weights: Optional[List[float]]) -> Dict[str, Any]:
        """Ensemble aggregation for classification results"""
        if len(results) == 1:
            return results[0]['outputs']
        
        # Combine outputs by averaging
        combined_outputs = {}
        
        # Get output names from first result
        output_names = list(results[0]['outputs'].keys())
        
        for output_name in output_names:
            output_arrays = []
            for result in results:
                if output_name in result['outputs']:
                    output_data = np.array(result['outputs'][output_name])
                    output_arrays.append(output_data)
            
            if output_arrays:
                # Average the outputs
                ensemble_output = np.mean(output_arrays, axis=0)
                combined_outputs[output_name] = ensemble_output.tolist()
        
        return combined_outputs
    
    async def _weighted_aggregation(self, results: List[Dict], weights: Optional[List[float]]) -> Dict[str, Any]:
        """Weighted aggregation based on provided weights"""
        if not weights or len(weights) != len(results):
            # Fallback to ensemble if weights are invalid
            return await self._ensemble_aggregation(results, weights)
        
        if len(results) == 1:
            return results[0]['outputs']
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        combined_outputs = {}
        output_names = list(results[0]['outputs'].keys())
        
        for output_name in output_names:
            weighted_sum = None
            
            for i, result in enumerate(results):
                if output_name in result['outputs']:
                    output_data = np.array(result['outputs'][output_name])
                    weighted_output = output_data * weights[i]
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_output
                    else:
                        weighted_sum += weighted_output
            
            if weighted_sum is not None:
                combined_outputs[output_name] = weighted_sum.tolist()
        
        return combined_outputs
    
    async def _confidence_based_aggregation(self, results: List[Dict], weights: Optional[List[float]]) -> Dict[str, Any]:
        """Aggregation based on confidence scores"""
        if len(results) == 1:
            return results[0]['outputs']
        
        # Extract confidence scores or use execution time as proxy
        confidences = []
        for result in results:
            metadata = result.get('metadata', {})
            # Use inverse of execution time as confidence proxy
            exec_time = metadata.get('execution_time_ms', 1000)
            confidence = 1.0 / (exec_time + 1)  # Avoid division by zero
            confidences.append(confidence)
        
        # Use confidences as weights
        return await self._weighted_aggregation(results, confidences)
    
    async def query_coordinator_status(self) -> Dict[str, Any]:
        """Query coordinator for status information"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{self.coordinator_url}/status")
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to query coordinator: {e}")
            return {"status": "unknown", "error": str(e)}

# Global agent instance
aggregator = AggregatorAgent(coordinator_url="inference-coordinator:8080")

@app.post("/aggregate")
async def aggregate_results(request: AggregationRequest):
    """Aggregate multiple inference results"""
    return await aggregator.aggregate_results(request)

@app.get("/health")
async def health():
    """Health check endpoint"""
    coordinator_status = await aggregator.query_coordinator_status()
    
    return {
        "status": "healthy",
        "agent": "aggregator",
        "coordinator_status": coordinator_status.get("status", "unknown"),
        "aggregation_strategies": list(aggregator.aggregation_strategies.keys())
    }

@app.get("/strategies")
async def list_strategies():
    """List available aggregation strategies"""
    return {
        "strategies": list(aggregator.aggregation_strategies.keys()),
        "descriptions": {
            "default": "Return first result or simple selection",
            "ensemble": "Average outputs across all results",
            "weighted": "Weighted combination based on provided weights",
            "confidence": "Weight by confidence scores or execution time"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)