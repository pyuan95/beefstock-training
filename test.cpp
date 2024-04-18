#include <algorithm>
#include <deque>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

using namespace chess;
using namespace binpack;

static Square orient(Color color, Square sq) {
    if (color == Color::White) {
        return sq;
    } else {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return sq.flippedVertically().flippedHorizontally();
    }
}

struct Transformer {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 13;  // 6 pieces for each side and empty square
    static constexpr int FEATURES_PER_SQ = NUM_SQ + NUM_PT;
    // Each square contains info about piece type and what squares it is attacking.
    static constexpr int INPUTS = FEATURES_PER_SQ * NUM_SQ;
    // max of 32 pieces, and each piece attacks at most 8 other pieces
    static constexpr int MAX_ACTIVE_FEATURES = 32 * 8 + 64;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color) {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = 0;
        for (Square sq : pieces) {
            // piece type feature
            auto p = pos.pieceAt(sq);
            int p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j++] = p_idx + ordinal(orient(color, sq)) * FEATURES_PER_SQ;

            // attacked/defended squares features
            auto attacked_pieces = pos.attacks(sq) & pieces;
            for (Square piece_sq : attacked_pieces) {
                values[j] = 1.0f;
                features[j++] = NUM_PT + ordinal(orient(color, piece_sq)) + ordinal(orient(color, sq)) * FEATURES_PER_SQ;
            }
            // consider adding features for pinned pieces
        }
        for (Square sq : EnumTraits<Square>::values) {
            if (!pieces.isSet(sq)) {
                values[j] = 1.0f;
                features[j++] = NUM_PT - 1 + ordinal(orient(color, sq)) * FEATURES_PER_SQ;
            }
        }

        return {j, INPUTS};
    }
};

int main() {
    Position p;
    p.set("8/3q1pk1/4pp2/2R4p/pP5P/b2NP1P1/5P1K/3R4 b - - 2 35");
    TrainingDataEntry e{p, Move(), 0, 0, 0};
    std::vector<int> features(Transformer::MAX_ACTIVE_FEATURES);
    std::vector<float> values(Transformer::MAX_ACTIVE_FEATURES);
    auto [j, _] = Transformer::fill_features_sparse(e, features.data(), values.data(), Color::Black);
    for (int i = 0; i < j; i++) {
        std::cout << features[i] << " ";
    }
    std::cout << "\n";
    return 0;
}