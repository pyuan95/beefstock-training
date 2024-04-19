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
NUM_ROWS = 8

FEATURES_PER_SQUARE = NUM_SQ * NUM_PT * 3
NUM_FEATURES = NUM_SQ * FEATURES_PER_SQUARE
MAX_ACTIVE_FEATURES = NUM_SQ + NUM_SQ * 8 + NUM_SQ * 8

static_movement_bonus = torch.zeros([1, 1, NUM_SQ, NUM_SQ])
for i in range(64):
    for j in range(64):
        dr, dc = i % NUM_ROWS - j % NUM_ROWS, i // NUM_ROWS - j // NUM_ROWS
        if i != j and (abs(dr) == abs(dc) or dr == 0 or dc == 0 or sorted([abs(dr), abs(dc)]) == [1, 2]):
            static_movement_bonus[:, :, i, j] = 1

assert (static_movement_bonus == static_movement_bonus.transpose(2, 3)).all()


def positional_encoding(seq_length: int, depth: int):
    result = torch.arange(0, seq_length).reshape([-1, 1]) / (
        torch.pow(10000, (torch.arange(0, depth) // 2 * 2.0) / depth)
    ).reshape([1, -1])
    result[:, torch.arange(0, depth, 2)] = torch.sin(result[:, torch.arange(0, depth, 2)])
    result[:, torch.arange(1, depth, 2)] = torch.cos(result[:, torch.arange(1, depth, 2)])
    return result[None, :, :]


def init_weight(*dims, dtype=torch.float32):
    stdv = 1.0 / (max(dims) ** 0.5)
    weight = torch.zeros(dims, dtype=dtype)
    weight.uniform_(-stdv, stdv)
    return weight


def potential_moves_attention(indices):
    """
    returns a matrix of shape N, NUM_SQ, NUM_SQ
    M[b, i, j] = 1 if piece on square i could attack square j if there are no squares in between
    transpose gives info about attacked squares instead of attacking squares
    very hacky, but dont worry about it it works
    """
    squares = attacking_squares = indices // FEATURES_PER_SQUARE
    intra_square_position = indices % FEATURES_PER_SQUARE
    piece_types = intra_square_position
    mask = (indices >= 0) & (intra_square_position < NUM_SQ * NUM_PT)


def movement_attention(indices):
    """
    returns a matrix of shape N, NUM_SQ, NUM_SQ
    M[b, i, j] = 1 if piece on square i attacks square j in batch b, 0 otherwise
    transpose gives info about attacked squares instead of attacking squares
    very hacky, but dont worry about it it works
    """
    N = indices.shape[0]
    intra_square_position = indices % FEATURES_PER_SQUARE
    mask = (indices >= 0) & (intra_square_position >= NUM_SQ * NUM_PT) & (intra_square_position < NUM_SQ * NUM_PT * 2)
    attacking_squares = indices // FEATURES_PER_SQUARE
    attacked_squares = (intra_square_position - NUM_SQ * NUM_PT) // NUM_PT
    attention_indices = attacking_squares * NUM_SQ + attacked_squares
    attention_indices[~mask] = NUM_SQ**2
    attention_indices = attention_indices.to(torch.int64)
    result = torch.zeros(N, NUM_SQ**2 + 1, device=indices.device)
    result.scatter_(1, attention_indices, 1.0)
    return result[:, : NUM_SQ**2].reshape(N, NUM_SQ, NUM_SQ)


# class RMSNorm(nn.Module):
#     def __init__(self, depth, eps=1e-7):
#         super().__init__()
#         self.g = nn.Parameter(1.0)
#         self.b = nn.Parameter(0.0)
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


class Encoder(nn.Module):
    def __init__(self, depth, n_heads, dff, eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            depth,
            n_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(depth, dff, bias=True),
            nn.ReLU(),
            nn.Linear(dff, depth, bias=True),
        )
        self.norm1 = RMSNorm(depth, eps=eps)
        self.norm2 = RMSNorm(depth, eps=eps)
        self.nsq = NUM_SQ  # for torch.jit.script

    def forward(self, x, attn_mask=None):
        # attacks: N, NUM_SQ, NUM_SQ
        N = x.shape[0]
        """
        attacks = attacks.unsqueeze(1)
        attacked = attacks.transpose(2, 3)
        attn_adjustments = (
            attacks * self.attacks_scale.exp2()
            + attacked * self.attacked_scale.exp2()
            + self.static_movement_bonus * self.static_scale.exp2()
        )
        attn_out, _ = self.self_attn(x, x, x, need_weights=False, attn_mask=attn_adjustments.reshape(-1, NUM_SQ, NUM_SQ))
        """

        attn_out, _ = self.self_attn(x, x, x, need_weights=False, attn_mask=attn_mask)
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
        smolgen_hidden=64,
        n_heads=8,
        gamma=0.95,
        start_lambda=0.0,
        max_epoch=400,
        end_lambda=0.0,
        lr=1e-3,
        eps=1e-6,
        activation_function="relu",
    ):
        super(Transformer, self).__init__()
        self.lr = lr
        self.start_lambda = start_lambda
        self.max_epoch = max_epoch
        self.end_lambda = end_lambda
        self.gamma = gamma
        self.depth = depth
        self.n_heads = n_heads
        self.nnue2score = 300.0
        act = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }[activation_function]

        self.smolgen = nn.Sequential(
            nn.Linear(depth * NUM_SQ, smolgen_hidden * 2),
            act,
            nn.Linear(smolgen_hidden * 2, smolgen_hidden),
            act,
            nn.Linear(smolgen_hidden, n_heads * NUM_SQ * NUM_SQ),
        )
        self.piece_encoder = FeatureTransformerSlice(FEATURES_PER_SQUARE, depth, NUM_SQ)
        self.piece_norm = RMSNorm(self.depth)
        self.encoders = nn.ModuleList([Encoder(depth, n_heads, dff) for _ in range(n_layers)])
        self.transformer_hidden = nn.Sequential(
            nn.Linear(depth * NUM_SQ * 2, eval_hidden),
            act,
            nn.Linear(eval_hidden, eval_hidden),
            act,
        )
        self.ffn_hidden = nn.Sequential(
            nn.Linear(depth * NUM_SQ * 2, eval_hidden),
            act,
            nn.Linear(eval_hidden, eval_hidden),
            act,
        )
        self.eval = nn.Sequential(
            nn.Linear(eval_hidden * 2, eval_hidden),
            act,
            nn.Linear(eval_hidden, 1),
        )
        self.init_weights()

    def init_weights(self):
        input_weights = self.material_weights.weight
        input_bias = self.material_weights.bias

        vals = [126, 781, 825, 1276, 2538]
        sc = 1 / self.nnue2score
        with torch.no_grad():
            # init psqt
            input_weights[:] = 0.0
            input_bias[:] = 0.0
            input_weights = input_weights.view(NUM_SQ, 3, NUM_SQ * NUM_PT, NUM_PSQT_BUCKETS)
            for i in range(5):
                input_weights[:, 0, torch.arange(i * 2, NUM_SQ * NUM_PT, NUM_PT), :] = -vals[i] * sc
                input_weights[:, 0, torch.arange(i * 2 + 1, NUM_SQ * NUM_PT, NUM_PT), :] = vals[i] * sc
                input_weights[:, 0, torch.arange(i * 2, NUM_SQ * NUM_PT, NUM_PT), :] = vals[i] * sc
                input_weights[:, 0, torch.arange(i * 2 + 1, NUM_SQ * NUM_PT, NUM_PT), :] = -vals[i] * sc

            # init piece encoder weights the same way
            input_weights = self.piece_encoder.weight.view(
                NUM_SQ, 3, NUM_SQ, NUM_PT, self.depth
            )  # ksq, psq/at/atr, sq, pt, depth
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

        # create piece embeddings and ffn embedding
        ones = torch.ones(white_indices.shape, device=self.device)
        white_emb = self.piece_norm(self.piece_encoder(white_indices, ones).reshape(N, NUM_SQ, self.depth))
        black_emb = self.piece_norm(self.piece_encoder(black_indices, ones).reshape(N, NUM_SQ, self.depth))
        ffn_emb = torch.cat([white_emb, black_emb], dim=1) * us + torch.cat([black_emb, white_emb], dim=1) * them
        ffn_emb = self.ffn_hidden(ffn_emb.reshape(N, -1))

        # white POV score
        white_smolgen = self.smolgen(white_emb.reshape(N, -1)).reshape(N * self.n_heads, NUM_SQ, NUM_SQ)
        for encoder in self.encoders:
            white_emb = encoder(white_emb, white_smolgen)
        white_emb = white_emb.reshape(N, -1)

        # black POV score
        black_smolgen = self.smolgen(black_emb.reshape(N, -1)).reshape(N * self.n_heads, NUM_SQ, NUM_SQ)
        for encoder in self.encoders:
            black_emb = encoder(black_emb, black_smolgen)
        black_emb = black_emb.reshape(N, -1)

        transformer_emb = torch.cat([white_emb, black_emb], dim=1) * us + torch.cat([black_emb, white_emb], dim=1) * them
        transformer_emb = self.transformer_hidden(transformer_emb)

        return self.eval(torch.cat([ffn_emb, transformer_emb], dim=1))

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
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())  # estimated match result (using win, loss and draw probs).

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
