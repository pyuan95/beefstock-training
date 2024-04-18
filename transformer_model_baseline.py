import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
import numpy as np

from feature_transformer import DoubleFeatureTransformerSlice
from feature_transformer_squares import FeatureTransformerSlice

torch.set_float32_matmul_precision("medium")

BOARD_SIZE = 8
NUM_SQ = 64
NUM_PT = 13
NUM_PSQT_BUCKETS = 8

FEATURES_PER_SQUARE = NUM_SQ * NUM_PT * 3
NUM_FEATURES = NUM_SQ * FEATURES_PER_SQUARE
MAX_ACTIVE_FEATURES = NUM_SQ + NUM_SQ * 8 + NUM_SQ * 8


def positional_encoding(seq_length: int, depth: int):
    result = torch.arange(0, seq_length).reshape([-1, 1]) / (torch.pow(10000, (torch.arange(0, depth) // 2 * 2.0) / depth)).reshape([1, -1])
    result[:, torch.arange(0, depth, 2)] = torch.sin(result[:, torch.arange(0, depth, 2)])
    result[:, torch.arange(1, depth, 2)] = torch.cos(result[:, torch.arange(1, depth, 2)])
    return result[None, :, :]


def init_weight(*dims, dtype=torch.float32):
    stdv = 1.0 / (max(dims) ** 0.5)
    weight = torch.zeros(dims, dtype=dtype)
    weight.uniform_(-stdv, stdv)
    return weight


# class RMSNorm(nn.Module):
#     def __init__(self, depth, eps=1e-7):
#         super().__init__()
#         self.g = nn.Parameter(torch.ones(depth))
#         self.b = nn.Parameter(torch.zeros(depth))
#         self.eps = eps

#     def forward(self, t):
#         rmse = torch.sqrt((t**2).sum(dim=-1, keepdim=True))
#         return t / (rmse + self.eps) * self.g + self.b


class RMSNorm(nn.Module):
    def __init__(self, depth, eps=1e-7):
        super().__init__()
        self.norm = nn.LayerNorm(depth)

    def forward(self, t):
        return self.norm(t)


class FirstEncoderWithAttentionAsParameters(nn.Module):
    """
    The first encoder layer.
    Attention values are parameters - thus no need to generate keys or queries
    didnt' work :(
    """

    def __init__(self, depth, dff, num_heads, eps=1e-5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert depth % num_heads == 0
        self.depth = depth
        self.num_heads = num_heads
        self.value_encoder = nn.Embedding(NUM_FEATURES, depth)
        # self.attn_weights = nn.Embedding(NUM_ATTENTION_WEIGHTS, num_heads)
        self.attn_bias = nn.Embedding(NUM_PT * NUM_PT, num_heads)
        self.attn_scale = nn.Parameter(torch.tensor(100.0))
        self.ffn = nn.Sequential(
            nn.Linear(depth, dff, bias=True),
            nn.ReLU(),
            nn.Linear(dff, depth, bias=True),
        )
        self.norm1 = RMSNorm(depth, eps=eps)
        self.norm2 = RMSNorm(depth, eps=eps)
        self.step = 0

    def forward(self, piece_indices, attn_bias_indices, ksq):
        # piece_indices: (N, num_sq)
        # attn_bias_indices: (N, num_sq * num_sq)
        # piece_indices are (ksq, psq, pt)
        N, _ = piece_indices.shape
        piece_indices_no_ksq = piece_indices - ksq.reshape(N, 1) * NUM_SQ * NUM_PT  # (N, num_sq)
        attn_indices = piece_indices_no_ksq.unsqueeze(2) * NUM_SQ * NUM_PT + piece_indices_no_ksq.unsqueeze(1)
        attn_indices += ksq.reshape(N, 1, 1) * NUM_SQ * NUM_PT * NUM_SQ * NUM_PT  # (N, num_sq, num_sq)
        attn_bias_indices = attn_bias_indices.reshape(N, NUM_SQ, NUM_SQ)

        # TODO: for some reason, attention movement bias slow down training to a crawl, figure out how to make it faster
        # attn = (self.attn_weights[attn_indices] + self.attn_bias[attn_bias_indices]).permute(0, 3, 1, 2)  # (N, num_heads, num_sq, num_sq)
        attn = (
            ((self.attn_weights(attn_indices) + self.attn_bias(attn_bias_indices.transpose(1, 2))) * self.attn_scale)
            .permute(0, 3, 1, 2)
            .softmax(dim=-1)
        )  # (N, num_heads, num_sq, num_sq)
        ######
        if self.step % 200 == 0:
            np.save("./debug/attn", attn.detach().cpu().numpy()[0:5, 0:5, :, :])
            np.save("./debug/pieces", piece_indices_no_ksq.detach().cpu().numpy()[0:5, :])
        self.step += 1
        ######
        values = (
            self.value_encoder(piece_indices).reshape(N, NUM_SQ, self.num_heads, self.depth // self.num_heads).permute(0, 2, 1, 3)
        )  # (N, num_heads, num_sq, depth // num_heads)
        output = self.norm1((attn @ values + values).transpose(1, 2).reshape(N, NUM_SQ, self.depth))
        return self.norm2(self.ffn(output) + output)


class FirstEncoder(nn.Module):
    """
    The first encoder layer.
    """

    def __init__(self, depth, dff, num_heads, eps=1e-5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert depth % num_heads == 0
        self.depth = depth
        self.num_heads = num_heads
        self.piece_encoder = FeatureTransformerSlice(FEATURES_PER_SQUARE, depth, NUM_SQ)
        # self.key_encoder = nn.Linear(depth, depth, bias=False)
        # self.query_encoder = nn.Linear(depth, depth, bias=False)
        # self.value_encoder = nn.Linear(depth, depth, bias=False)
        # self.attn_attack_bias = nn.Embedding(NUM_PT * NUM_PT, num_heads)
        # self.attn_attacked_bias = nn.Embedding(NUM_PT * NUM_PT, num_heads)
        # self.attn_bias_scale = nn.Parameter(torch.tensor(100.0))

        # self.mha = nn.MultiheadAttention(
        #     depth,
        #     num_heads,
        #     dropout=0.0,
        #     batch_first=True,
        # )
        # self.ffn = nn.Sequential(
        #     nn.Linear(depth, dff, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(dff, depth, bias=True),
        # )
        # self.norm0 = RMSNorm(depth, eps=eps)
        # self.norm1 = RMSNorm(depth, eps=eps)
        # self.norm2 = RMSNorm(depth, eps=eps)
        self.step = 0

    def forward(self, piece_indices, piece_values):
        x = self.piece_encoder(piece_indices, piece_values)
        return x


class Encoder(nn.Module):
    def __init__(self, depth, n_heads, dff, eps=1e-5):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            depth,
            n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.queries = nn.Linear(depth, depth, bias=False)
        self.keys = nn.Linear(depth, depth, bias=False)
        self.values = nn.Linear(depth, depth, bias=False)
        self.projection = nn.Linear(depth, depth, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(depth, dff, bias=True),
            nn.ReLU(),
            nn.Linear(dff, depth, bias=True),
        )

        self.norm1 = RMSNorm(depth, eps=eps)
        self.norm2 = RMSNorm(depth, eps=eps)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        attn_out, _ = self.attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )

        x1 = self.norm1(x + attn_out)
        x2 = self.norm2(self.ffn(x1) + x1)
        return x2


class Transformer(pl.LightningModule):
    """
    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores

    gamma - the multiplicative factor applied to the learning rate after each epoch

    lr - the initial learning rate
    """

    def __init__(
        self,
        n_layers=3,
        depth=64,
        dff=96,
        eval_hidden=128,
        n_heads=1,
        gamma=0.95,
        start_lambda=0.0,
        max_epoch=400,
        end_lambda=0.0,
        lr=1e-3,
        eps=1e-6,
    ):
        super(Transformer, self).__init__()
        self.lr = lr
        self.start_lambda = start_lambda
        self.max_epoch = max_epoch
        self.end_lambda = end_lambda
        self.gamma = gamma
        self.depth = depth
        self.nnue2score = 300.0

        self.piece_norm = nn.LayerNorm(self.depth)
        self.piece_encoder = FeatureTransformerSlice(FEATURES_PER_SQUARE, depth, NUM_SQ)
        self.material_weights = DoubleFeatureTransformerSlice(NUM_FEATURES, NUM_PSQT_BUCKETS)
        self.encoders = nn.ModuleList(
            [nn.TransformerEncoderLayer(depth, n_heads, dff, dropout=0.0, batch_first=True) for _ in range(n_layers)]
        )
        self.evals = nn.Sequential(
            nn.Linear(depth * NUM_SQ * 2, eval_hidden),
            nn.ReLU(),
            nn.Linear(eval_hidden, eval_hidden),
            nn.ReLU(),
            nn.Linear(eval_hidden, 1),
        )
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # init psqt
            input_weights = self.material_weights.weight
            input_bias = self.material_weights.bias
            vals = [126, 781, 825, 1276, 2538]
            sc = 1 / self.nnue2score
            input_weights[:] = 0.0
            input_bias[:] = 0.0
            input_weights = input_weights.view(NUM_SQ, 3, NUM_SQ * NUM_PT, NUM_PSQT_BUCKETS)
            for i in range(5):
                input_weights[:, 0, torch.arange(i * 2, NUM_SQ * NUM_PT, NUM_PT), :] = vals[i] * sc
                input_weights[:, 0, torch.arange(i * 2 + 1, NUM_SQ * NUM_PT, NUM_PT), :] = -vals[i] * sc

            # init piece encoder weights the same way
            input_weights = self.piece_encoder.weight.view(NUM_SQ, 3, NUM_SQ, NUM_PT, self.depth)  # ksq, psq/at/atr, sq, pt, depth
            for i in range(3):
                input_weights[:, i : i + 1, :, :, :] = input_weights[:1, i : i + 1, :1, :, :]

    def forward(
        self,
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        psqt_indices,
    ):
        (N, _) = us.shape

        # material count
        wm, bm = self.material_weights(white_indices, white_values, black_indices, black_values)
        wm = wm.gather(1, psqt_indices.unsqueeze(1))
        bm = bm.gather(1, psqt_indices.unsqueeze(1))
        material_contribution = (wm - bm) * (us - 0.5) * self.nnue2score

        ones = torch.ones(white_values.shape, device=white_values.device, dtype=wm.dtype)
        # white POV score
        white_emb = self.piece_norm(self.piece_encoder(white_indices, ones).reshape(N, NUM_SQ, self.depth))
        for encoder in self.encoders:
            white_emb = encoder(white_emb)
        white_emb = white_emb.reshape(N, -1)

        # black POV score
        black_emb = self.piece_norm(self.piece_encoder(black_indices, ones).reshape(N, NUM_SQ, self.depth))
        for encoder in self.encoders:
            black_emb = encoder(black_emb)
        black_emb = black_emb.reshape(N, -1)

        emb = torch.cat([white_emb, black_emb], dim=1) * us + torch.cat([black_emb, white_emb], dim=1) * them

        eval_contribution = self.evals(emb) * self.nnue2score
        return material_contribution + eval_contribution, eval_contribution, material_contribution

    def step_(self, batch, batch_idx, loss_type):
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        in_scaling = 340
        out_scaling = 380
        offset = 270
        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch
        scorenet, ec, mc = self(us, them, white_indices, white_values, black_indices, black_values, psqt_indices)
        q = (scorenet - offset) / in_scaling  # used to compute the chance of a win
        qm = (-scorenet - offset) / in_scaling  # used to compute the chance of a loss
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())  # estimated match result
        # qf = scorenet.sigmoid()

        p = (score - offset) / out_scaling
        pm = (-score - offset) / out_scaling
        pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())

        t = outcome
        actual_lambda = self.start_lambda + (self.end_lambda - self.start_lambda) * (self.current_epoch / self.max_epoch)
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        loss = torch.pow(torch.abs(pt - qf), 2.5).mean()

        self.log(loss_type, loss)
        self.log("MAE", torch.abs(pt - qf).mean())
        self.log("material contribution", mc.abs().mean())
        self.log("eval contribution", ec.abs().mean())
        self.log("average score", score.abs().clip(-1000, 1000).mean())

        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return [optimizer], [scheduler]
