// Performance benchmarks for the cube solver.
// Uses ParallelFastIDASolver (all three pattern DBs, 18 threads at root level).
// Run from the project root: ./build-release/benchmark --data-dir data
#include "../src/cube.h"
#include "../src/moves.h"
#include "../src/pattern_db.h"
#include "../src/fast_solver.h"
#include "../src/utils.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstring>

struct BenchResult {
    int    solution_length;  // -1 if not found
    uint64_t nodes;
    double seconds;
};

static BenchResult run_one(ParallelFastIDASolver& solver,
                            const std::vector<int>& scramble_moves) {
    CubeState state = apply_moves(SOLVED_CUBE, scramble_moves);
    auto r = solver.solve(state, 20);
    return {r.found ? (int)r.moves.size() : -1, r.nodes_explored, r.time_seconds};
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
            data_dir = argv[++i];
    }

    init_moves();

    PatternDatabases dbs;
    dbs.load_or_build(data_dir);
    build_cp_move_table();
    build_co_move_table();
    build_eo_move_table();
    build_inv_ep_table();

    ParallelFastIDASolver solver(dbs.corner_db, dbs.edge_db1, dbs.edge_db2);

    // Fixed scrambles at known optimal depths for reproducibility
    struct Case { const char* label; const char* scramble; };
    const std::vector<Case> fixed_cases = {
        {"4 moves",  "R U R' U'"},
        {"8 moves",  "R U R' U' F2 D' L B"},
        {"10 moves", "R U R' U' R' F R2 U' R' U"},
        {"12 moves", "R U2 R' U' R U' R' L' U2 L U L' U L"},
        {"15 moves", "R U2 R' F2 R U R' U' F' U' F R2 U' R' U"},
    };

    std::cout << "=== Fixed scrambles ===\n";
    std::cout << std::left
              << std::setw(12) << "Label"
              << std::setw(8)  << "Sol"
              << std::setw(16) << "Nodes"
              << "Time(s)\n"
              << std::string(50, '-') << "\n";

    for (const auto& c : fixed_cases) {
        auto moves = parse_move_sequence(c.scramble);
        auto r = run_one(solver, moves);
        std::cout << std::left
                  << std::setw(12) << c.label
                  << std::setw(8)  << (r.solution_length >= 0 ? std::to_string(r.solution_length) : "FAIL")
                  << std::setw(16) << r.nodes
                  << std::fixed << std::setprecision(4) << r.seconds << "\n";
    }

    // Random scrambles at several depths — 5 seeds each for representative spread
    const int SEEDS = 5;
    std::cout << "\n=== Random scrambles (5 seeds each) ===\n";
    std::cout << std::left
              << std::setw(8)  << "Depth"
              << std::setw(8)  << "Sol"
              << std::setw(16) << "Nodes"
              << std::setw(12) << "Time(s)"
              << "Scramble\n"
              << std::string(80, '-') << "\n";

    for (int depth : {8, 10, 12, 15}) {
        std::vector<double> times;
        for (int seed = 1; seed <= SEEDS; seed++) {
            auto scramble = random_scramble(depth, (unsigned)seed);
            auto r = run_one(solver, scramble);
            times.push_back(r.seconds);
            std::cout << std::left
                      << std::setw(8)  << ("d=" + std::to_string(depth))
                      << std::setw(8)  << (r.solution_length >= 0 ? std::to_string(r.solution_length) : "FAIL")
                      << std::setw(16) << r.nodes
                      << std::setw(12) << std::fixed << std::setprecision(4) << r.seconds
                      << format_move_sequence(scramble) << "\n";
        }
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double med = times[SEEDS / 2];
        std::sort(times.begin(), times.end());
        std::cout << "  depth=" << depth
                  << " avg=" << std::fixed << std::setprecision(3) << avg << "s"
                  << " median=" << times[SEEDS / 2] << "s"
                  << " min=" << times.front() << "s"
                  << " max=" << times.back() << "s\n\n";
        (void)med;
    }

    return 0;
}
