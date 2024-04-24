import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, depth, eps=1e-7):
        super().__init__()
        self.norm = nn.LayerNorm(depth)

    def forward(self, t):
        return self.norm(t)


@torch.jit.script
def scaled_dot_product_attention(keys, queries, values, attention_mask, num_heads):
    # assumes keys and values are of shape N, S, D, and queries N, L, D.
    N, S, D = keys.shape
    _, L, _ = queries.shape
    keys = keys.reshape(N, S, num_heads, D // num_heads).transpose(-2, -3)
    queries = queries.reshape(N, L, num_heads, D // num_heads).transpose(-2, -3)
    values = values.reshape(N, S, num_heads, D // num_heads).transpose(-2, -3)
    scale = 1 / torch.sqrt(keys.shape[-1])
    attn_weights = queries @ keys.transpose(-1, -2) * scale + attention_mask
    attn_weights = attn_weights.softmax(-1)
    output = attn_weights @ values  # N, ..., num_heads, L, D // num_heads
    output = output.transpose(-2, -3)  # N, ..., L, num_heads, D // num_heads
    output = output.flatten(-2, -1)
    return output, attn_weights


class Encoder(nn.Module):
    def __init__(self, depth_in, dff, depth_out, activation, eps=1e-5):
        super().__init__()
        self.d_in = depth_in
        self.d_out = depth_out
        self.kqv = nn.Linear(depth_in, depth_in * 3, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(depth_in, dff, bias=True),
            activation,
            nn.Linear(dff, depth_out, bias=True),
        )
        self.norm = RMSNorm(depth_out, eps=eps)

    def forward(self, x, attn_mask, num_heads):
        """
        shape of x: N, L, D_in
        shape of attn_mask: N, num_heads, L, L
        returns: tensor of shape N, L, D_out
        """
        N, L, _ = x.shape
        k, q, v = self.kqv(x).split(self.d_in, dim=-1)
        output, _ = scaled_dot_product_attention(k, q, v, attn_mask, num_heads)
        output = self.norm(self.ffn(output))
        x = x.reshape(N, L, -1, self.d_out).mean(-2)
        return x + output


encoder = Encoder(256, 128, 128, nn.ReLU())
num_heads = torch.tensor(8)
x = torch.rand(4, 4, 256)
attn_mask = torch.eye(4) + torch.zeros(1, 1, 1, 1)
print(encoder(x, attn_mask, num_heads).shape)
