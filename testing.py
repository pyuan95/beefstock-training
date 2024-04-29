import torch
import torch.nn as nn


@torch.jit.script
def batch_mul_ref(W, B, X):
    # @ operator broadcasts and uses too much memory
    # W shape: num_heads, D_out, D_in
    # B shape: num_heads, D_out
    # X shape: batch size, num_heads, D_in
    # returns: W @ X + B of shape batch size, num_heads, D_out

    return (W @ X.unsqueeze(-1)).squeeze(-1) + B


@torch.jit.script
def batch_mul(W, B, X):
    # @ operator broadcasts and uses too much memory
    # W shape: num_heads, D_out, D_in
    # B shape: num_heads, D_out
    # X shape: batch size, num_heads, D_in
    # returns: W @ X + B of shape batch size, num_heads, D_out
    x = torch.einsum("hij, bhj->bhi", W, X)
    x += B
    return x


from time import time

W = torch.randn(16, 4096, 64) * 100
B = torch.randn(16, 4096) * 100
X = torch.randn(1024, 16, 64) * 100

start = time()
for _ in range(100):
    print(_)
    cur = batch_mul(W, B, X)
    # print((W @ X.unsqueeze(-1)).shape)
print(time() - start)
