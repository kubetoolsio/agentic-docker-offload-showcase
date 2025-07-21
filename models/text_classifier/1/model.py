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
