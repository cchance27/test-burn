import torch;
import time;

def benchmark():
    device = torch.device('mps')  # Or 'cuda' if available; use 'cpu' for CPU comparison
    batch, seq, dim = 32, 1024, 64
    query = torch.randn(batch, seq, dim, device=device)
    key = torch.randn(batch, seq, dim, device=device)
    value = torch.randn(batch, seq, dim, device=device)
    iterations = 500
    
    torch.mps.synchronize()  # Warm-up (for MPS; use torch.cuda.synchronize() for CUDA)
    start = torch.mps.current_allocated_memory()  # Optional: monitor memory if needed
    t0 = time.time()
    for _ in range(iterations):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.mps.synchronize()  # Or appropriate sync
    t1 = time.time()
    print(f"PyTorch time for {iterations} iterations: {t1 - t0} seconds")

if __name__ == "__main__":
    benchmark()
    benchmark()