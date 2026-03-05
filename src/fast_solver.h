#pragma once
#include "cube.h"
#include "moves.h"
#include "pattern_db.h"
#include "solver.h"
#include <array>
#include <vector>

// Precomputed move tables for right multiplication.
//
// Right-mult formula:
//   new_cp[i] = cp[mv.cp[i]]
//   new_co[i] = (co[mv.cp[i]] + mv.co[i]) % 3   <- depends only on co, not cp
//   new_ep[i] = ep[mv.ep[i]]
//   new_eo[i] = (eo[mv.ep[i]] + mv.eo[i]) % 2   <- depends only on eo, not ep
//
// The co and eo transformations are independent of the current permutation,
// so CO_MOVE_TABLE and EO_MOVE_TABLE can be standalone index-to-index tables.
//
// Sizes:
//   CP: 18 × 40320 × 4 bytes ≈ 2.8 MB
//   CO: 18 × 2187  × 2 bytes ≈  79 KB
//   EO: 18 × 2048  × 2 bytes ≈  74 KB

using CpMoveTable = std::array<std::array<uint32_t, 40320>, NUM_MOVES>;
using CoMoveTable = std::array<std::array<uint16_t, 2187>,  NUM_MOVES>;
using EoMoveTable = std::array<std::array<uint16_t, 2048>,  NUM_MOVES>;

extern CpMoveTable CP_MOVE_TABLE;
extern CoMoveTable CO_MOVE_TABLE;
extern EoMoveTable EO_MOVE_TABLE;

// Call once after init_moves().
void build_cp_move_table();
void build_co_move_table();
void build_eo_move_table();

// IDA* solver that avoids CubeState copies and Lehmer re-encoding on every node.
//
// Bottlenecks in the naive solver:
//   1. compose(): allocates and fills a new CubeState (~80 ops per node)
//   2. encode_corner_perm(): Lehmer encoding is O(n²) (~100 ops per heuristic call)
//   3. encode_corner_orient() / encode_edge_orient() in the heuristic (~18 ops)
//
// This solver instead:
//   - Keeps state as raw arrays, modified in-place with save/restore
//   - Updates cp_idx, co_idx, eo_idx in O(1) via the move tables above
//   - heuristic() is two table lookups with no encoding
//
// Requires CornerPatternDB and EdgeOrientDB to be loaded.
// Call build_cp_move_table(), build_co_move_table(), build_eo_move_table() first.
//
// Uses right multiplication internally. The path is reversed before returning
// so the reported move sequence is correct for the standard left-mult convention.
class FastIDASolver {
public:
    FastIDASolver(const CornerPatternDB& corner_db,
                  const EdgeOrientDB&    edge_orient_db);

    SolveResult solve(const CubeState& start, int max_depth = 20);

    uint64_t get_nodes_explored() const { return nodes_explored_; }

private:
    const CornerPatternDB& corner_db_;
    const EdgeOrientDB&    edge_orient_db_;

    uint64_t         nodes_explored_;
    std::vector<int> path_;

    // Search state, modified in-place, saved/restored per recursion level (52 bytes)
    uint8_t  cp_[8], co_[8], ep_[12], eo_[12];
    uint32_t cp_idx_;  // Lehmer index of cp_, kept in sync by apply()
    uint32_t co_idx_;  // base-3 index of co_, kept in sync by apply()
    uint32_t eo_idx_;  // base-2 index of eo_, kept in sync by apply()

    static constexpr int FOUND = -1;
    static constexpr int INF   = 1000;

    int  heuristic() const;
    bool is_solved()  const;
    void apply(int m);

    int search(int g, int threshold, int prev_move);
};
