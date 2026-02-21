#include "solver.h"
#include <chrono>
#include <algorithm>
#include <cassert>
#include <iostream>

// ============================================================
// IDASolver implementation
// ============================================================

IDASolver::IDASolver(HeuristicFn heuristic)
    : heuristic_(std::move(heuristic)), nodes_explored_(0) {}

SolveResult IDASolver::solve(const CubeState& start, int max_depth) {
    auto t_start = std::chrono::high_resolution_clock::now();
    nodes_explored_ = 0;
    path_.clear();

    if (start.is_solved()) {
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        return SolveResult{true, {}, 0, elapsed};
    }

    int threshold = heuristic_(start);

    while (threshold <= max_depth) {
        path_.clear();
        int result = search(start, 0, threshold);

        if (result == FOUND) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_end - t_start).count();
            return SolveResult{true, path_, nodes_explored_, elapsed};
        }

        if (result == INF) {
            // No solution exists (shouldn't happen for valid cube)
            break;
        }

        threshold = result;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    return SolveResult{false, {}, nodes_explored_, elapsed};
}

int IDASolver::search(const CubeState& state, int g, int threshold) {
    int h = heuristic_(state);
    int f = g + h;

    if (f > threshold) return f;

    nodes_explored_++;

    if (state.is_solved()) return FOUND;

    int min_threshold = INF;
    int prev_move = path_.empty() ? -1 : path_.back();

    for (int m = 0; m < NUM_MOVES; m++) {
        // Pruning: skip redundant move sequences
        if (is_redundant_sequence(prev_move, m)) continue;

        CubeState next = apply_move(state, m);
        path_.push_back(m);

        int result = search(next, g + 1, threshold);

        if (result == FOUND) return FOUND;

        if (result < min_threshold) min_threshold = result;

        path_.pop_back();
    }

    return min_threshold;
}

// ============================================================
// Simple heuristic: misplaced cubies (admissible)
// Each move cycles exactly 4 corners and 4 edges.
// Therefore: at most 4 corners can be placed correctly per move,
// and at most 4 edges can be placed correctly per move.
// Taking max of the two separate bounds gives an admissible heuristic.
// ============================================================
int heuristic_misplaced(const CubeState& state) {
    int misplaced_corners = 0;
    for (int i = 0; i < NUM_CORNERS; i++) {
        if (state.cp[i] != i || state.co[i] != 0) misplaced_corners++;
    }
    int misplaced_edges = 0;
    for (int i = 0; i < NUM_EDGES; i++) {
        if (state.ep[i] != i || state.eo[i] != 0) misplaced_edges++;
    }
    // Each move fixes at most 4 corners and at most 4 edges
    int h_corners = (misplaced_corners + 3) / 4;
    int h_edges   = (misplaced_edges   + 3) / 4;
    return std::max(h_corners, h_edges);
}

