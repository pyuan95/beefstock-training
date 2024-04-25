import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from feature_transformer_squares import FeatureTransformerSlice

NUM_SQ = 64
NUM_PT = 13
NUM_MOVES = 64
FEATURES_PER_SQUARE = NUM_SQ * NUM_PT * 3


def accuracy(predictions, labels):
    return (predictions.argmax(-1) == labels).float().mean()


class RMSNorm(nn.Module):
    def __init__(self, depth, eps=1e-7):
        super().__init__()
        self.norm = nn.LayerNorm(depth)

    def forward(self, t):
        return self.norm(t)


class DenseSquares(pl.LightningModule):
    def __init__(
        self,
        depth=256,
        eval_hidden=128,
        gamma=0.95,
        start_lambda=0.0,
        max_epoch=400,
        end_lambda=0.0,
        lr=1e-3,
        policy_classification_weight=0.0001,
        activation_function="relu",
    ):
        super(DenseSquares, self).__init__()
        self.lr = lr
        self.start_lambda = start_lambda
        self.max_epoch = max_epoch
        self.end_lambda = end_lambda
        self.gamma = gamma
        self.policy_classification_weight = policy_classification_weight
        activation = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }[activation_function]

        self.depth = depth
        self.piece_encoder = FeatureTransformerSlice(FEATURES_PER_SQUARE, depth, NUM_SQ)
        self.piece_norm = RMSNorm(depth)
        self.ffn_hidden = nn.Sequential(
            nn.Linear(depth * NUM_SQ * 2, eval_hidden * 4),
            activation,
            nn.Linear(eval_hidden * 4, eval_hidden * 2),
            activation,
            nn.Linear(eval_hidden * 2, eval_hidden),
            activation,
        )
        self.final_eval = nn.Sequential(
            nn.Linear(eval_hidden, eval_hidden),
            activation,
            nn.Linear(eval_hidden, 1),
        )
        self.outcome_classification = nn.Sequential(
            nn.Linear(eval_hidden, eval_hidden),
            activation,
            nn.Linear(eval_hidden, 3),
        )
        self.policy_classification = nn.Sequential(
            nn.Linear(eval_hidden, eval_hidden),
            activation,
            nn.Linear(eval_hidden, NUM_SQ * NUM_MOVES),
        )
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # init piece encoder weights the same way
            # ksq, psq/at/atr, sq, pt, depth
            input_weights = self.piece_encoder.weight.view(NUM_SQ, 3, NUM_SQ, NUM_PT, self.depth)
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
        white_emb = self.piece_norm(self.piece_encoder(white_indices, ones).reshape(N, NUM_SQ, self.depth))
        black_emb = self.piece_norm(self.piece_encoder(black_indices, ones).reshape(N, NUM_SQ, self.depth))
        ffn_emb = torch.cat([white_emb, black_emb], dim=1) * us.reshape(N, 1, 1) + torch.cat([black_emb, white_emb], dim=1) * them.reshape(
            N, 1, 1
        )
        emb = self.ffn_hidden(ffn_emb.reshape(N, -1))
        return (
            self.final_eval(emb),
            self.outcome_classification(emb.detach()),
            self.policy_classification(emb).reshape(N, -1),
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
        self.log(loss_type, loss, sync_dist=True)
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
