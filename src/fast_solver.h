#pragma once
#include "cube.h"
#include "moves.h"
#include "pattern_db.h"
#include "solver.h"
#include <array>
#include <atomic>
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

// inv_ep[m][p] = destination of the edge at position p after move m (right mult).
// Used to track partial edge positions incrementally.
using InvEpTable = std::array<std::array<uint8_t, 12>, NUM_MOVES>;
extern InvEpTable INV_EP_TABLE;
void build_inv_ep_table();

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
//   - Tracks partial edge state (pos[6], ori[6]) for each group incrementally
//   - Caches ep1_idx_, ep2_idx_ (recomputed in apply(), not heuristic())
//   - heuristic() is three table lookups with no encoding
//
// Requires all three pattern DBs. Call build_cp/co/eo_move_table() and
// build_inv_ep_table() before constructing.
//
// Uses right multiplication internally; path reversed before returning.
class FastIDASolver {
public:
    static constexpr int FOUND = -1;
    static constexpr int INF   = 1000;

    FastIDASolver(const CornerPatternDB& corner_db,
                  const EdgePatternDB&   edge_db1,
                  const EdgePatternDB&   edge_db2,
                  const EdgePatternDB&   edge_db3);

    SolveResult solve(const CubeState& start, int max_depth = 20);

    uint64_t get_nodes_explored() const { return nodes_explored_; }

    // Compute h(start) without starting a search.
    int compute_heuristic(const CubeState& start);

    // Run one IDA* threshold restricted to root_move only.
    // Checks *found each node for early exit. Returns FOUND or next threshold.
    // If FOUND, path() holds the right-mult sequence; caller must reverse it.
    int search_threshold_single(const CubeState& start, int root_move,
                                int threshold, const std::atomic<bool>& found);

    const std::vector<int>& path() const { return path_; }

private:
    const CornerPatternDB& corner_db_;
    const EdgePatternDB&   edge_db1_;
    const EdgePatternDB&   edge_db2_;
    const EdgePatternDB&   edge_db3_;

    uint64_t         nodes_explored_;
    std::vector<int> path_;

    // Set by search_threshold_single for early exit when another thread finds a solution.
    const std::atomic<bool>* abort_ = nullptr;

    // Core state - cp/co/ep/eo arrays + encoded indices for corner and full EO
    uint8_t  cp_[8], co_[8], ep_[12], eo_[12];
    uint32_t cp_idx_;   // Lehmer index of cp_
    uint32_t co_idx_;   // base-3 index of co_
    uint32_t eo_idx_;   // base-2 index of eo_ (kept for is_solved())

    // Partial edge tracking for edge pattern DBs (groups 0, 1, 2)
    // ep1_pos_[k] = position of edge label k     (group 0: labels 0-5)
    // ep2_pos_[k] = position of edge label 6+k   (group 1: labels 6-11)
    // ep3_pos_[k] = position of edge label G2[k] (group 2: labels 0,2,4,6,8,10)
    uint8_t ep1_pos_[6], ep1_ori_[6];
    uint8_t ep2_pos_[6], ep2_ori_[6];
    uint8_t ep3_pos_[6], ep3_ori_[6];
    uint32_t ep1_idx_;  // encode_edge_partial result for group 0, kept in sync
    uint32_t ep2_idx_;  // encode_edge_partial result for group 1, kept in sync
    uint32_t ep3_idx_;  // encode_edge_partial result for group 2, kept in sync

    void init_state(const CubeState& start);
    int  heuristic() const;
    bool is_solved()  const;
    void apply(int m);
    uint32_t encode_ep(const uint8_t pos[6], const uint8_t ori[6]) const;

    int search(int g, int threshold, int prev_move);
};

// Parallel IDA* solver. Within each threshold iteration, distributes the
// 18 root moves across threads so they run simultaneously. Optimal: all
// threads search the same threshold, so the first FOUND result is guaranteed
// to be a minimum-move solution.
//
// Same preconditions as FastIDASolver (build all move tables first).
class ParallelFastIDASolver {
public:
    ParallelFastIDASolver(const CornerPatternDB& corner_db,
                          const EdgePatternDB&   edge_db1,
                          const EdgePatternDB&   edge_db2,
                          const EdgePatternDB&   edge_db3);

    SolveResult solve(const CubeState& start, int max_depth = 20);

    uint64_t get_nodes_explored() const { return nodes_explored_; }

private:
    const CornerPatternDB& corner_db_;
    const EdgePatternDB&   edge_db1_;
    const EdgePatternDB&   edge_db2_;
    const EdgePatternDB&   edge_db3_;
    uint64_t nodes_explored_;
};
