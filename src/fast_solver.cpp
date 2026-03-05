#include "fast_solver.h"
#include <chrono>
#include <cstring>
#include <algorithm>

CpMoveTable CP_MOVE_TABLE;
CoMoveTable CO_MOVE_TABLE;
EoMoveTable EO_MOVE_TABLE;

// Corner permutation move table (right multiplication).
// new_cp[i] = cp[mv.cp[i]]
// CP_MOVE_TABLE[m][cp_idx] = encode_corner_perm(new_cp)
void build_cp_move_table() {
    for (int m = 0; m < NUM_MOVES; m++) {
        const CubeState& mv = MOVE_TABLE[m];
        for (uint32_t idx = 0; idx < 40320; idx++) {
            auto cp = decode_corner_perm(idx);
            std::array<uint8_t, NUM_CORNERS> ncp;
            for (int i = 0; i < NUM_CORNERS; i++)
                ncp[i] = cp[mv.cp[i]];
            CP_MOVE_TABLE[m][idx] = encode_corner_perm(ncp);
        }
    }
}

// Corner orientation move table (right multiplication).
// new_co[i] = (co[mv.cp[i]] + mv.co[i]) % 3
// Independent of the current corner permutation, so a standalone table is valid.
// CO_MOVE_TABLE[m][co_idx] = encode_corner_orient(new_co)
void build_co_move_table() {
    for (int m = 0; m < NUM_MOVES; m++) {
        const CubeState& mv = MOVE_TABLE[m];
        for (uint32_t idx = 0; idx < 2187; idx++) {
            auto co = decode_corner_orient(idx);
            std::array<uint8_t, NUM_CORNERS> nco;
            for (int i = 0; i < NUM_CORNERS; i++)
                nco[i] = (co[mv.cp[i]] + mv.co[i]) % 3;
            CO_MOVE_TABLE[m][idx] = (uint16_t)encode_corner_orient(nco);
        }
    }
}

// Edge orientation move table (right multiplication).
// new_eo[i] = (eo[mv.ep[i]] + mv.eo[i]) % 2
// Independent of the current edge permutation, so a standalone table is valid.
// EO_MOVE_TABLE[m][eo_idx] = encode_edge_orient(new_eo)
void build_eo_move_table() {
    for (int m = 0; m < NUM_MOVES; m++) {
        const CubeState& mv = MOVE_TABLE[m];
        for (uint32_t idx = 0; idx < 2048; idx++) {
            auto eo = decode_edge_orient(idx);
            std::array<uint8_t, NUM_EDGES> neo;
            for (int i = 0; i < NUM_EDGES; i++)
                neo[i] = (eo[mv.ep[i]] + mv.eo[i]) % 2;
            EO_MOVE_TABLE[m][idx] = (uint16_t)encode_edge_orient(neo);
        }
    }
}

FastIDASolver::FastIDASolver(const CornerPatternDB& corner_db,
                             const EdgeOrientDB&    edge_orient_db)
    : corner_db_(corner_db), edge_orient_db_(edge_orient_db),
      nodes_explored_(0), cp_idx_(0), co_idx_(0), eo_idx_(0)
{
    memset(cp_, 0, sizeof cp_);
    memset(co_, 0, sizeof co_);
    memset(ep_, 0, sizeof ep_);
    memset(eo_, 0, sizeof eo_);
}

SolveResult FastIDASolver::solve(const CubeState& start, int max_depth) {
    auto t0 = std::chrono::high_resolution_clock::now();
    nodes_explored_ = 0;
    path_.clear();

    // Load initial state; encode indices once, reuse across threshold iterations
    for (int i = 0; i < 8;  i++) { cp_[i] = start.cp[i]; co_[i] = start.co[i]; }
    for (int i = 0; i < 12; i++) { ep_[i] = start.ep[i]; eo_[i] = start.eo[i]; }
    const uint32_t start_cp_idx = encode_corner_perm(start.cp);
    const uint32_t start_co_idx = encode_corner_orient(start.co);
    const uint32_t start_eo_idx = encode_edge_orient(start.eo);
    cp_idx_ = start_cp_idx;
    co_idx_ = start_co_idx;
    eo_idx_ = start_eo_idx;

    if (is_solved()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        return {true, {}, 0, std::chrono::duration<double>(t1 - t0).count()};
    }

    int threshold = heuristic();

    while (threshold <= max_depth) {
        for (int i = 0; i < 8;  i++) { cp_[i] = start.cp[i]; co_[i] = start.co[i]; }
        for (int i = 0; i < 12; i++) { ep_[i] = start.ep[i]; eo_[i] = start.eo[i]; }
        cp_idx_ = start_cp_idx;
        co_idx_ = start_co_idx;
        eo_idx_ = start_eo_idx;
        path_.clear();

        int result = search(0, threshold, -1);

        if (result == FOUND) {
            // Right-mult search produces path [m_1, ..., m_k] with
            // initial ∘ m_1 ∘ ... ∘ m_k = identity, i.e. m_1...m_k = initial^{-1}.
            // Physical solution (left-mult convention): [m_k, ..., m_1].
            std::reverse(path_.begin(), path_.end());
            auto t1 = std::chrono::high_resolution_clock::now();
            return {true, path_, nodes_explored_,
                    std::chrono::duration<double>(t1 - t0).count()};
        }
        if (result == INF) break;
        threshold = result;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return {false, {}, nodes_explored_,
            std::chrono::duration<double>(t1 - t0).count()};
}

// Two table lookups, no encoding.
int FastIDASolver::heuristic() const {
    int h = (int)corner_db_.lookup_idx(cp_idx_ * 2187u + co_idx_);
    return std::max(h, (int)edge_orient_db_.lookup_idx(eo_idx_));
}

bool FastIDASolver::is_solved() const {
    if (cp_idx_ != 0) return false;
    for (int i = 0; i < 8;  i++) if (co_[i] != 0) return false;
    for (int i = 0; i < 12; i++) if (ep_[i] != (uint8_t)i || eo_[i] != 0) return false;
    return true;
}

// Apply move m in-place.  Right multiplication: compose(state, mv).
//   ncp[i] = cp[mv.cp[i]],   nco[i] = (co[mv.cp[i]] + mv.co[i]) % 3
//   nep[i] = ep[mv.ep[i]],   neo[i] = (eo[mv.ep[i]] + mv.eo[i]) % 2
// All three cached indices updated via the precomputed tables.
void FastIDASolver::apply(int m) {
    const CubeState& mv = MOVE_TABLE[m];
    uint8_t ncp[8], nco[8], nep[12], neo[12];

    for (int i = 0; i < 8; i++) {
        ncp[i] = cp_[mv.cp[i]];
        nco[i] = (co_[mv.cp[i]] + mv.co[i]) % 3;
    }
    for (int i = 0; i < 12; i++) {
        nep[i] = ep_[mv.ep[i]];
        neo[i] = (eo_[mv.ep[i]] + mv.eo[i]) % 2;
    }

    cp_idx_ = CP_MOVE_TABLE[m][cp_idx_];
    co_idx_ = CO_MOVE_TABLE[m][co_idx_];
    eo_idx_ = EO_MOVE_TABLE[m][eo_idx_];
    memcpy(cp_, ncp, 8);  memcpy(co_, nco, 8);
    memcpy(ep_, nep, 12); memcpy(eo_, neo, 12);
}

int FastIDASolver::search(int g, int threshold, int prev_move) {
    int h = heuristic();
    int f = g + h;
    if (f > threshold) return f;
    if (is_solved()) return FOUND;

    nodes_explored_++;

    // Save state (52 bytes, fits in one cache line)
    uint8_t  scp[8], sco[8], sep[12], seo[12];
    uint32_t scp_idx = cp_idx_, sco_idx = co_idx_, seo_idx = eo_idx_;
    memcpy(scp, cp_, 8);  memcpy(sco, co_, 8);
    memcpy(sep, ep_, 12); memcpy(seo, eo_, 12);

    int min_t = INF;

    for (int m = 0; m < NUM_MOVES; m++) {
        if (is_redundant_sequence(prev_move, m)) continue;

        apply(m);
        path_.push_back(m);

        int result = search(g + 1, threshold, m);
        if (result == FOUND) return FOUND;
        if (result < min_t) min_t = result;

        path_.pop_back();

        cp_idx_ = scp_idx; co_idx_ = sco_idx; eo_idx_ = seo_idx;
        memcpy(cp_, scp, 8);  memcpy(co_, sco, 8);
        memcpy(ep_, sep, 12); memcpy(eo_, seo, 12);
    }

    return min_t;
}
