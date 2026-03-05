#pragma once
#include "cube.h"
#include "moves.h"
#include <vector>
#include <functional>
#include <cstdint>

// IDA* (Iterative Deepening A*): memory-efficient optimal search.
// Runs iterative depth-first search with increasing f-cost thresholds.
// f(n) = g(n) + h(n) where g = moves from start, h = admissible lower bound.
// Optimal whenever h is admissible (never overestimates).

struct SolveResult {
    bool found;
    std::vector<int> moves;  // solution sequence
    uint64_t nodes_explored;
    double time_seconds;
};

using HeuristicFn = std::function<int(const CubeState&)>;

class IDASolver {
public:
    explicit IDASolver(HeuristicFn heuristic);

    SolveResult solve(const CubeState& start, int max_depth = 20);

    uint64_t get_nodes_explored() const { return nodes_explored_; }

private:
    HeuristicFn heuristic_;
    uint64_t    nodes_explored_;
    std::vector<int> path_;

    static constexpr int FOUND = -1;
    static constexpr int INF   = 1000;

    int search(const CubeState& state, int g, int threshold);
};

// Admissible heuristic: max(misplaced_corners/4, misplaced_edges/4).
// Weak but requires no precomputed data; useful for testing.
int heuristic_misplaced(const CubeState& state);
