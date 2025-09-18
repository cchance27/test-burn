import torch

def verify(batch, seq, dim):
    query = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)
    key = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)
    value = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)

    output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)

    print("PyTorch SDPA output:", output.flatten().tolist())


def verify2(batch, seq, dim):
    query = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)
    key = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)
    value = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)

    output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)

    print("PyTorch SDPA output:", output.flatten().tolist())

if __name__ == "__main__":
    verify(8, 1024, 8)
    verify2(8, 1024, 8)