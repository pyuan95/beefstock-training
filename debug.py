import numpy as np
import os
import torch

NUM_SQ = 64
FEATURES_PER_SQUARE = 64 * 4 + 13

files = [
    "nan_output",
    "indices",
    "us",
    "them",
    "white_indices",
    "white_values",
    "black_indices",
    "black_values",
    "outcome",
    "score",
    "inp",
]

arrs = []

for f in files:
    arrs.append(np.load(os.path.join("debug", f + ".npy")))

(
    nan_output,
    indices,
    us,
    them,
    white_indices,
    white_values,
    black_indices,
    black_values,
    outcome,
    score,
    inp,
) = arrs


def index2piece(i):
    pieces = ["P", "N", "B", "R", "Q", "K", " "]
    p = pieces[i // 2]
    return p.lower() if i % 2 == 1 else p


def coord(*args):
    if len(args) == 2:
        i, j = args
    else:
        t = args[0]
        i, j = t // 8, t % 8

    return "abcdefgh"[j] + str(i + 1)


def print_board_from_indices(idxs):
    board = [[None for _ in range(8)] for __ in range(8)]
    misc = []
    for i in idxs:
        if i < 0:
            break
        sq = i // FEATURES_PER_SQUARE
        i = i % FEATURES_PER_SQUARE
        if i < 13:
            p = index2piece(i)
            board[sq // 8][sq % 8] = p
        else:
            i -= 13
            attacker = "White" if i < 128 else "Black"
            typ = "defends" if i < 64 or i >= 64 * 3 else "attacks"
            other_sq = i % 64
            misc.append(f"{attacker} at {coord(sq)} {typ} {coord(other_sq)}")
    for row in reversed(board):
        for c in row:
            print(c, end="|")
        print()
    print(misc)


def print_board(b):
    b = b.reshape(8, 8, FEATURES_PER_SQUARE)
    attacks = [[] for _ in range(64)]
    fen = [[] for _ in range(8)]
    for i in range(7, -1, -1):
        for j in range(8):
            p = index2piece(list(b[i, j]).index(1))
            print(p, end="|")
            if p != " ":
                fen[i].append(p)
            else:
                if fen[i] and type(fen[i][-1]) == int:
                    fen[i][-1] += 1
                else:
                    fen[i].append(1)

            v = list(b[i, j][13:])
            while sum(v) > 0:
                idx = v.index(1)
                attacks[i * 8 + j].append(coord(idx // 8, idx % 8))
                v[idx] = 0
        print()
    fen = "/".join(["".join([str(x) for x in z]) for z in reversed(fen)]) + " w - - 0 0"
    print("fen:", fen)
    for i in range(64):
        if attacks[i] and False:
            print(f"attacks from {coord(i // 8, i % 8)}: {attacks[i]}")

            # for indices, white_to_play, result in zip(white_indices, us, outcome):
            #     b = np.zeros(NUM_SQ * FEATURES_PER_SQUARE)
            #     for index in indices:
            #         b[index] = 1.0
            #     b = b.reshape(8, 8, FEATURES_PER_SQUARE)
            #     print_board(b)
            #     print("white to play", white_to_play)
            #     print("result", result)
            #     input()


for row in nan_output[0].reshape(64, -1):
    print([str(z)[:4] for z in row.round(2).tolist()])
exit()
################################

for board in indices.astype(np.int32):
    print_board_from_indices(board)
    input()

#########################################

out_scaling = 380
offset = 270

score = torch.tensor(score)

p = (score - offset) / out_scaling
pm = (-score - offset) / out_scaling
pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())
pf = pf.numpy()
score = score.numpy()

for board, white_to_play, sc, adj_sc, res in zip(inp, us, score, pf, outcome):
    print_board(board)
    print("is it white to play?", white_to_play)
    print("result", res)
    print("score", sc)
    print("adjusted score", adj_sc)
    input()
