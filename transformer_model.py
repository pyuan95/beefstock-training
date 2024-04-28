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
NUM_MOVES = 64

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


def accuracy(predictions, labels):
    return (predictions.argmax(-1) == labels).float().mean()


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


# class Encoder(nn.Module):
#     def __init__(self, depth, n_heads, dff, eps=1e-5):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             depth,
#             n_heads,
#             dropout=0.0,
#             batch_first=True,
#         )

#         self.ffn = nn.Sequential(
#             nn.Linear(depth, dff, bias=True),
#             nn.ReLU(),
#             nn.Linear(dff, depth, bias=True),
#         )
#         self.norm1 = RMSNorm(depth, eps=eps)
#         self.norm2 = RMSNorm(depth, eps=eps)
#         self.nsq = NUM_SQ  # for torch.jit.script

#     def forward(self, x, attn_mask=None):
#         # attacks: N, NUM_SQ, NUM_SQ
#         N = x.shape[0]
#         attn_out, _ = self.self_attn(x, x, x, need_weights=False, attn_mask=attn_mask)
#         x1 = self.norm1(x + attn_out)
#         x2 = self.norm2(self.ffn(x1) + x1)
#         return x2


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


class Smolgen(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, activation):
        super().__init__()
        self.activation = activation
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, num_heads * hidden_dim),
            activation,
            nn.Linear(num_heads * hidden_dim, num_heads * hidden_dim),
            activation,
        )
        self.out = nn.Conv1d(1, NUM_SQ * NUM_SQ, hidden_dim, stride=hidden_dim)

    def forward(self, emb):
        N = emb.shape[0]
        emb = self.hidden(emb).reshape(N, 1, -1)
        emb = self.out(emb)  # (N, NUM_SQ * NUM_SQ, n_heads)
        return emb.transpose(1, 2).reshape(N, self.num_heads, NUM_SQ, NUM_SQ)


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
        depth_list=[64],
        dff_list=[64],
        eval_hidden=128,
        smolgen_hidden=64,
        n_heads=8,
        gamma=0.95,
        start_lambda=0.0,
        max_epoch=400,
        end_lambda=0.0,
        lr=1e-3,
        policy_classification_weight=0.01,
        activation_function="relu",
    ):
        super(Transformer, self).__init__()
        self.lr = lr
        self.start_lambda = start_lambda
        self.max_epoch = max_epoch
        self.end_lambda = end_lambda
        self.gamma = gamma
        self.n_heads = torch.tensor(n_heads)
        self.policy_classification_weight = policy_classification_weight
        self.nnue2score = 300.0
        activation = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }[activation_function]

        initial_depth = depth_list[0]
        final_depth = depth_list[-1]
        self.initial_depth = initial_depth
        self.final_depth = final_depth
        self.smolgen = torch.compile(Smolgen(initial_depth * NUM_SQ, smolgen_hidden, n_heads, activation))
        self.piece_encoder = FeatureTransformerSlice(FEATURES_PER_SQUARE, initial_depth, NUM_SQ)
        self.piece_norm = RMSNorm(initial_depth)
        self.encoders = nn.ModuleList(
            [Encoder(depth_list[i], dff_list[i], depth_list[i + 1], activation) for i in range(len(depth_list) - 1)]
        )
        self.transformer_hidden = nn.Sequential(
            nn.Linear(final_depth * NUM_SQ * 2, eval_hidden),
            activation,
            nn.Linear(eval_hidden, eval_hidden),
            activation,
        )
        self.ffn_hidden = nn.Sequential(
            nn.Linear(initial_depth * NUM_SQ * 2, eval_hidden * 2),
            activation,
            nn.Linear(eval_hidden * 2, eval_hidden),
            activation,
        )
        self.final_eval = nn.Sequential(
            nn.Linear(eval_hidden * 2, eval_hidden),
            activation,
            nn.Linear(eval_hidden, 1),
        )
        self.outcome_classification = nn.Sequential(
            nn.Linear(eval_hidden * 2, eval_hidden),
            activation,
            nn.Linear(eval_hidden, 3),
        )
        self.policy_classification = nn.Sequential(
            nn.Linear(final_depth, final_depth),
            activation,
            nn.Linear(final_depth, NUM_MOVES),
        )
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # init piece encoder weights the same way
            # ksq, psq/at/atr, sq, pt, depth
            input_weights = self.piece_encoder.weight.view(NUM_SQ, 3, NUM_SQ, NUM_PT, self.initial_depth)
            for i in range(3):
                input_weights[:, i : i + 1, :, :, :] = input_weights[:1, i : i + 1, :1, :, :]

    def forward(
        self,
        us,
        them,
        white_indices,
        black_indices,
    ):
        N = us.shape[0]

        # create piece embeddings and ffn embedding
        ones = torch.ones(white_indices.shape, device=self.device)
        white_emb = self.piece_norm(self.piece_encoder(white_indices, ones).reshape(N, NUM_SQ, self.initial_depth))
        black_emb = self.piece_norm(self.piece_encoder(black_indices, ones).reshape(N, NUM_SQ, self.initial_depth))
        ffn_emb = torch.cat([white_emb, black_emb], dim=1) * us.reshape(N, 1, 1) + torch.cat([black_emb, white_emb], dim=1) * them.reshape(
            N, 1, 1
        )
        ffn_emb = self.ffn_hidden(ffn_emb.reshape(N, -1))

        # white POV score
        white_smolgen = self.smolgen(white_emb.reshape(N, -1))
        for encoder in self.encoders:
            white_emb = encoder(white_emb, white_smolgen, self.n_heads)
        white_emb = white_emb.reshape(N, -1)

        # black POV score
        black_smolgen = self.smolgen(black_emb.reshape(N, -1))
        for encoder in self.encoders:
            black_emb = encoder(black_emb, black_smolgen, self.n_heads)
        black_emb = black_emb.reshape(N, -1)

        side_to_play_emb = (white_emb * us + black_emb * them).reshape(N, NUM_SQ, self.final_depth)

        transformer_emb = torch.cat([white_emb, black_emb], dim=1) * us + torch.cat([black_emb, white_emb], dim=1) * them
        transformer_emb = self.transformer_hidden(transformer_emb)

        emb = torch.cat([ffn_emb, transformer_emb], dim=1)

        return (
            self.final_eval(emb),
            self.outcome_classification(emb.detach()),
            self.policy_classification(side_to_play_emb).reshape(N, -1),
        )

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
            policy_index,
            psqt_indices,
            layer_stack_indices,
        ) = batch
        scorenet, outcome_pred, policy_pred = self(us, them, white_indices, black_indices)
        # q = (scorenet - offset) / in_scaling  # used to compute the chance of a win
        # qm = (-scorenet - offset) / in_scaling  # used to compute the chance of a loss
        # qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())  # estimated match result (using win, loss and draw probs).
        qf = scorenet.sigmoid()

        p = (score - offset) / out_scaling
        pm = (-score - offset) / out_scaling
        pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())

        t = outcome
        actual_lambda = self.start_lambda + (self.end_lambda - self.start_lambda) * (self.current_epoch / self.max_epoch)
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # regression loss
        regression_loss = torch.pow(torch.abs(pt - qf), 2.5).mean()

        # outcome classification loss
        outcome_true = (outcome * 2).to(torch.int64).reshape(-1)
        outcome_classification_loss = F.cross_entropy(outcome_pred, outcome_true)

        # policy classifivation loss
        policy_classification_loss = F.cross_entropy(policy_pred, policy_index.reshape(-1))

        loss = regression_loss + outcome_classification_loss + policy_classification_loss * self.policy_classification_weight
        self.log(loss_type, loss)
        self.log("regression loss", regression_loss, sync_dist=True)
        self.log("outcome accuracy", accuracy(outcome_pred, outcome_true), sync_dist=True)
        self.log("policy accuracy", accuracy(policy_pred, policy_index), sync_dist=True)
        self.log("MAE", torch.abs(pt - qf).mean(), sync_dist=True)
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
