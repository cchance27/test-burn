#!/usr/bin/env python3
"""
Script to implement the complete SwiGLU as used in Qwen2.5 model with projections
for comparison with Rust implementation.

Based on the architecture from QWEN2.5_ARCHITECTURE.md:
- gate_proj: Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
- up_proj: Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
- down_proj: Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)
- SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
"""
import torch
import torch.nn.functional as F
import numpy as np
import json

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def swiglu_with_projections(x, gate_weight, up_weight, down_weight):
    """
    Full SwiGLU implementation as used in Qwen2.5:
    SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    
    Args:
        x: Input tensor of shape (..., HIDDEN_SIZE)
        gate_weight: Weight matrix for gate projection (HIDDEN_SIZE x INTERMEDIATE_SIZE)
        up_weight: Weight matrix for up projection (HIDDEN_SIZE x INTERMEDIATE_SIZE)
        down_weight: Weight matrix for down projection (INTERMEDIATE_SIZE x HIDDEN_SIZE)
    
    Returns:
        Output tensor of shape (..., HIDDEN_SIZE)
    """
    # Apply gate and up projections using linear transformation
    gate_proj = F.linear(x, gate_weight)  # Shape: (..., INTERMEDIATE_SIZE)
    up_proj = F.linear(x, up_weight)      # Shape: (..., INTERMEDIATE_SIZE)
    
    # Apply SwiGLU: SiLU(gate_proj) * up_proj
    swiglu_result = F.silu(gate_proj) * up_proj  # Shape: (..., INTERMEDIATE_SIZE)
    
    # Apply final down projection
    output = F.linear(swiglu_result, down_weight)  # Shape: (..., HIDDEN_SIZE)
    
    return output


def generate_qwen25_parameters():
    """Generate parameters matching Qwen2.5 specifications."""
    HIDDEN_SIZE = 896
    INTERMEDIATE_SIZE = 4864
    
    # Generate weights similar to what might be in a trained model
    # Using Xavier/Glorot initialization (simplified)
    gate_weight = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE) * (2.0 / (HIDDEN_SIZE + INTERMEDIATE_SIZE)) ** 0.5
    up_weight = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE) * (2.0 / (HIDDEN_SIZE + INTERMEDIATE_SIZE)) ** 0.5
    down_weight = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE) * (2.0 / (INTERMEDIATE_SIZE + HIDDEN_SIZE)) ** 0.5
    
    return gate_weight, up_weight, down_weight


def generate_test_inputs():
    """Generate test inputs for the full SwiGLU implementation."""
    HIDDEN_SIZE = 896
    
    inputs = [
        torch.randn(1, HIDDEN_SIZE),      # Single sequence
        torch.randn(2, HIDDEN_SIZE),      # Batch size 2
        torch.randn(1, 1, HIDDEN_SIZE),   # With sequence dimension
        torch.randn(2, 3, HIDDEN_SIZE),   # Batch and sequence
    ]
    
    # Add some edge cases with smaller dimensions for easier verification
    inputs.extend([
        torch.zeros(1, HIDDEN_SIZE),
        torch.ones(1, HIDDEN_SIZE),
        torch.full((1, HIDDEN_SIZE), -1.0),
        torch.full((1, HIDDEN_SIZE), 0.5),
    ])
    
    return inputs


def dump_full_swiglu_comparison_data():
    """Generate and dump full SwiGLU comparison data."""
    gate_weight, up_weight, down_weight = generate_qwen25_parameters()
    
    results = []
    
    test_inputs = generate_test_inputs()
    for i, input_tensor in enumerate(test_inputs):
        try:
            output = swiglu_with_projections(input_tensor, gate_weight, up_weight, down_weight)
            
            # Save FULL input and output flattened for exact comparison
            # Also save full weights once (large, but reproducible)
            result = {
                'test_id': i,
                'input_shape': list(input_tensor.shape),
                'input': input_tensor.flatten().tolist(),  # Full flattened input
                'output_shape': list(output.shape),
                'output': output.flatten().tolist(),       # Full flattened output
                'name': f'full_swiglu_test_{i:02d}',
                'parameter_shapes': {
                    'gate_weight': list(gate_weight.shape),
                    'up_weight': list(up_weight.shape),
                    'down_weight': list(down_weight.shape),
                }
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing input {i}: {e}")
    
    # Save results to JSON file
    with open('swiglu_full_qwen25_comparison_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated {len(results)} full SwiGLU test cases with seeded data")
    print("Results saved to swiglu_full_qwen25_comparison_data.json")
    
    # Save full weights separately for Rust loading (once, shared across tests)
    weights = {
        'gate_weight': gate_weight.flatten().tolist(),
        'up_weight': up_weight.flatten().tolist(),
        'down_weight': down_weight.flatten().tolist(),
        'shapes': {
            'gate_weight': list(gate_weight.shape),
            'up_weight': list(up_weight.shape),
            'down_weight': list(down_weight.shape),
        }
    }
    
    with open('swiglu_qwen25_weights_full.json', 'w') as f:
        json.dump(weights, f, indent=2)
    print("Full weights saved to swiglu_qwen25_weights_full.json")


def verify_full_swiglu():
    """Verify the full SwiGLU implementation with simple values."""
    print("\nVerifying full SwiGLU implementation...")
    
    # Use a much smaller example for verification
    small_hidden = 4
    small_intermediate = 8
    
    # Create small weights for verification
    gate_w = torch.ones(small_intermediate, small_hidden) * 0.1
    up_w = torch.ones(small_intermediate, small_hidden) * 0.2
    down_w = torch.ones(small_hidden, small_intermediate) * 0.3
    
    # Simple input
    x = torch.ones(1, small_hidden) * 2.0
    
    result = swiglu_with_projections(x, gate_w, up_w, down_w)
    print(f"Verification test - Input: {x.flatten()}, Output: {result.flatten()}")


if __name__ == "__main__":
    dump_full_swiglu_comparison_data()
    verify_full_swiglu()