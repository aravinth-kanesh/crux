#include "cube.h"
#include "moves.h"
#include "solver.h"
#include "pattern_db.h"
#include "fast_solver.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <filesystem>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "\nOptions:\n"
              << "  --scramble \"R U R' ...\"   Solve the given scramble\n"
              << "  --random-depth N           Solve a random N-move scramble\n"
              << "  --seed N                   RNG seed for random scramble (default: random)\n"
              << "  --data-dir DIR             Pattern DB directory (default: ./data)\n"
              << "  --no-pattern-db            Use simple heuristic (slow, for testing)\n"
              << "  --max-depth N              Maximum solution depth (default: 20)\n"
              << "  --interactive              Enter moves interactively\n"
              << "  --verify SOLUTION          Verify a solution against a scramble\n"
              << "  --help                     Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " --scramble \"R U R' U'\"\n"
              << "  " << prog << " --random-depth 15\n"
              << "  " << prog << " --random-depth 18 --seed 42\n";
}

static void run_interactive(std::function<SolveResult(const CubeState&, int)> solve_fn, int max_depth) {
    std::cout << "Interactive mode. Enter moves (e.g. R U R' U'), 'solve' to solve, 'quit' to exit.\n";
    CubeState state;
    std::string line;

    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;

        if (line == "quit" || line == "exit") break;

        if (line == "solve") {
            if (state.is_solved()) {
                std::cout << "Already solved!\n";
                continue;
            }
            std::cout << "Solving...\n";
            auto result = solve_fn(state, max_depth);
            std::cout << format_solve_stats(result) << "\n";
            continue;
        }

        if (line == "reset") {
            state = SOLVED_CUBE;
            std::cout << "Reset to solved state.\n";
            continue;
        }

        if (line == "show") {
            print_cube_ascii(state);
            continue;
        }

        // Try to parse as move sequence
        try {
            auto moves = parse_move_sequence(line);
            state = apply_moves(state, moves);
            std::cout << "Applied " << moves.size() << " move(s). "
                      << (state.is_solved() ? "SOLVED!" : "Not solved.") << "\n";
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
            std::cout << "Commands: solve, reset, show, quit\n";
            std::cout << "Moves: U U' U2 D D' D2 F F' F2 B B' B2 L L' L2 R R' R2\n";
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string scramble_str;
    int random_depth = 0;
    unsigned rng_seed = 0;
    std::string data_dir = "data";
    bool no_pattern_db = false;
    int max_depth = 20;
    bool interactive = false;
    std::string verify_solution;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--scramble") == 0 && i + 1 < argc) {
            scramble_str = argv[++i];
        } else if (strcmp(argv[i], "--random-depth") == 0 && i + 1 < argc) {
            random_depth = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            rng_seed = (unsigned)std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--no-pattern-db") == 0) {
            no_pattern_db = true;
        } else if (strcmp(argv[i], "--max-depth") == 0 && i + 1 < argc) {
            max_depth = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--interactive") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) {
            verify_solution = argv[++i];
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Initialise move tables
    init_moves();

    // Set up solver: ParallelFastIDASolver when pattern DBs are available, IDASolver otherwise.
    PatternDatabases dbs;
    std::unique_ptr<ParallelFastIDASolver> fast_solver;
    std::unique_ptr<IDASolver>             ida_solver;
    std::function<SolveResult(const CubeState&, int)> solve_fn;

    if (!no_pattern_db) {
        std::filesystem::create_directories(data_dir);
        dbs.load_or_build(data_dir);
        build_cp_move_table();
        build_co_move_table();
        build_eo_move_table();
        build_inv_ep_table();
        fast_solver = std::make_unique<ParallelFastIDASolver>(dbs.corner_db, dbs.edge_db1, dbs.edge_db2, dbs.edge_db3);
        solve_fn = [&](const CubeState& s, int d) { return fast_solver->solve(s, d); };
    } else {
        std::cerr << "Using simple misplaced-cubies heuristic (slow!)\n";
        ida_solver = std::make_unique<IDASolver>(HeuristicFn(heuristic_misplaced));
        solve_fn = [&](const CubeState& s, int d) { return ida_solver->solve(s, d); };
    }

    // Interactive mode
    if (interactive) {
        run_interactive(solve_fn, max_depth);
        return 0;
    }

    // Determine scramble
    CubeState state;
    std::vector<int> scramble_moves;

    if (!scramble_str.empty()) {
        try {
            scramble_moves = parse_move_sequence(scramble_str);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing scramble: " << e.what() << "\n";
            return 1;
        }
    } else if (random_depth > 0) {
        scramble_moves = random_scramble(random_depth, rng_seed);
        std::cout << "Scramble (" << random_depth << " moves): "
                  << format_move_sequence(scramble_moves) << "\n";
    } else {
        // No scramble given
        print_usage(argv[0]);
        return 1;
    }

    // Apply scramble
    state = apply_moves(SOLVED_CUBE, scramble_moves);

    // Validate
    std::string err;
    if (!validate_cube(state, err)) {
        std::cerr << "Invalid cube state: " << err << "\n";
        return 1;
    }

    if (state.is_solved()) {
        std::cout << "Already solved (0 moves).\n";
        return 0;
    }

    // Show scrambled state
    std::cout << "Scrambled:\n";
    print_cube_ascii(state);

    // Solve
    std::cout << "Solving...\n";
    auto result = solve_fn(state, max_depth);

    std::cout << format_solve_stats(result) << "\n";

    // Verify solution if requested or always
    if (result.found) {
        CubeState check = apply_moves(state, result.moves);
        if (!check.is_solved()) {
            std::cerr << "BUG: Solution does not solve the cube!\n";
            return 1;
        }

        // Print scramble + solution when available
        if (!scramble_moves.empty()) {
            std::cout << "Scramble:  " << format_move_sequence(scramble_moves) << "\n";
            std::cout << "Solution:  " << format_move_sequence(result.moves) << "\n";
        }

        std::cout << "Solved:\n";
        print_cube_ascii(check);
    }

    if (!verify_solution.empty()) {
        try {
            auto sol_moves = parse_move_sequence(verify_solution);
            CubeState check = apply_moves(state, sol_moves);
            if (check.is_solved()) {
                std::cout << "Verification: Solution is correct (" << sol_moves.size() << " moves)\n";
            } else {
                std::cout << "Verification: Solution does NOT solve the cube\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing verification solution: " << e.what() << "\n";
        }
    }

    return result.found ? 0 : 1;
}
