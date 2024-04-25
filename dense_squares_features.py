import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT = 13
NUM_FEATURES = (NUM_SQ * 4 + NUM_PT) * NUM_SQ


def orient(is_white_pov: bool, sq: int):
    return (56 * (not is_white_pov)) ^ sq


class Features(FeatureBlock):
    def __init__(self):
        super(Features, self).__init__("DenseSquares", 0x5F134CBA, OrderedDict([("DenseSquares", NUM_FEATURES)]))

    def get_active_features(self, board: chess.Board):
        raise Exception("Not supported yet, you must use the c++ data loader for support during training")

    def get_initial_psqt_features(self):
        raise Exception("Not supported yet for DenseSquares")


"""
This is used by the features module for discovery of feature blocks.
"""


def get_feature_block_clss():
    return [Features]
