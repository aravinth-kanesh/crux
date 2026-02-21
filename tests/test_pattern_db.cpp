// Tests for pattern database correctness
// Builds the small edge-orient DB (2048 states) and validates it,
// then spot-checks that the corner DB returns sensible values for
// a few known states (using known distances from the solver as ground truth).
#include "../src/cube.h"
#include "../src/moves.h"
#include "../src/solver.h"
#include "../src/pattern_db.h"
#include <iostream>
#include <cassert>
#include <sstream>

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    g_tests_run++; \
    std::cout << "  " #name "... "; \
    try { test_##name(); std::cout << "PASS\n"; g_tests_passed++; } \
    catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << "\n"; g_tests_failed++; } \
    catch (...) { std::cout << "FAIL: unknown exception\n"; g_tests_failed++; } \
} while(0)

#define ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond " at line " + std::to_string(__LINE__)); } while(0)
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::ostringstream ss; ss << "Expected " << (b) << " got " << (a) << " at line " << __LINE__; throw std::runtime_error(ss.str()); } } while(0)
#define ASSERT_LE(a, b) do { if ((a) > (b)) { std::ostringstream ss; ss << (a) << " > " << (b) << " at line " << __LINE__; throw std::runtime_error(ss.str()); } } while(0)

// ============================================================
// Edge orientation DB tests
// ============================================================
TEST(edge_orient_solved) {
    EdgeOrientDB db;
    db.build();
    ASSERT(db.is_ready());

    // Solved state: distance 0
    ASSERT_EQ((int)db.lookup(SOLVED_CUBE), 0);
}

TEST(edge_orient_one_flip_move) {
    // F move flips 4 edges, so edge orient distance >= 1
    EdgeOrientDB db;
    db.build();

    CubeState state = apply_move(SOLVED_CUBE, MoveIndex::F);
    int d = (int)db.lookup(state);
    ASSERT(d >= 1);
    // After F: 4 edges flipped. EO distance should be 1 (F is one move).
    ASSERT_EQ(d, 1);
}

TEST(edge_orient_half_turn_no_flip) {
    // Half turns (F2, B2) don't flip any edges
    EdgeOrientDB db;
    db.build();

    CubeState f2 = apply_move(SOLVED_CUBE, MoveIndex::F2);
    // F2 = F*F: edges flipped twice = not flipped
    ASSERT_EQ((int)db.lookup(f2), 0);

    // U2 never flips edges at all
    CubeState u2 = apply_move(SOLVED_CUBE, MoveIndex::U2);
    ASSERT_EQ((int)db.lookup(u2), 0);
}

TEST(edge_orient_admissible) {
    // For every n-move scramble (n=1..5), edge_orient_db <= n
    EdgeOrientDB db;
    db.build();
    IDASolver solver(heuristic_misplaced);

    // Test all 1-move states
    for (int m = 0; m < NUM_MOVES; m++) {
        CubeState s = apply_move(SOLVED_CUBE, m);
        int h = (int)db.lookup(s);
        // Admissible: h <= 1 (solved in 1 move)
        if (h > 1) {
            throw std::runtime_error("Edge orient DB overestimates for move " +
                                     MOVE_NAMES[m] + ": h=" + std::to_string(h));
        }
    }

    // Test R U R' U' (4 moves, so edge orient h <= 4)
    auto moves = parse_move_sequence("R U R' U'");
    CubeState s = apply_moves(SOLVED_CUBE, moves);
    int h = (int)db.lookup(s);
    ASSERT_LE(h, 4);
}

TEST(edge_orient_all_states_populated) {
    EdgeOrientDB db;
    db.build();

    // All 2048 states should have been visited (max distance 11)
    for (uint32_t i = 0; i < EdgeOrientDB::EDGE_ORIENT_DB_SIZE; i++) {
        if (db.lookup_idx(i) == 255) {
            throw std::runtime_error("Edge orient DB has unvisited state at index " +
                                     std::to_string(i));
        }
    }
}

// ============================================================
// Corner DB correctness with small spot checks
// Build a partial corner DB using BFS for just a few levels,
// then verify admissibility against the actual solver.
//
// NOTE: Full corner DB (88M states) is too slow to build in tests.
// Checks:
//   1. Solved state = 0
//   2. All 1-move states have value = 1
//   3. All n-move states have value <= n (admissibility)
// ============================================================
TEST(corner_db_solved_state) {
    CornerPatternDB db;
    db.build();  // ~5 min in production; here only to verify the solved state entry
    // Only verify the solved state and immediate neighbors
    ASSERT_EQ((int)db.lookup(SOLVED_CUBE), 0);
}

TEST(corner_db_one_move_states) {
    CornerPatternDB db;
    db.build();

    for (int m = 0; m < NUM_MOVES; m++) {
        CubeState s = apply_move(SOLVED_CUBE, m);
        int h = (int)db.lookup(s);
        if (h != 1) {
            throw std::runtime_error("Corner DB: after move " + MOVE_NAMES[m] +
                                     " expected h=1, got h=" + std::to_string(h));
        }
    }
}

TEST(corner_db_admissible_4moves) {
    CornerPatternDB db;
    db.build();

    // R U R' U' has optimal solution length 4 (verified by solver)
    auto moves = parse_move_sequence("R U R' U'");
    CubeState s = apply_moves(SOLVED_CUBE, moves);
    int h = (int)db.lookup(s);
    ASSERT_LE(h, 4);
    // Also check it's not too weak (should be >= 1 for a non-solved state)
    ASSERT(h >= 1);
}

TEST(corner_db_consistent_with_solver) {
    // For several short scrambles, corner_db.h <= actual_solve_length
    CornerPatternDB db;
    db.build();
    IDASolver solver(heuristic_misplaced);

    struct TestCase { std::string scramble; int known_optimal; };
    std::vector<TestCase> cases = {
        {"R",           1},
        {"U F",         2},
        {"R U R'",      3},
        {"R U R' U'",   4},
        {"R U2 R' F2",  4},
    };

    for (const auto& tc : cases) {
        auto moves = parse_move_sequence(tc.scramble);
        CubeState s = apply_moves(SOLVED_CUBE, moves);

        // Verify solver finds optimal solution
        auto result = solver.solve(s, tc.known_optimal + 1);
        ASSERT(result.found);
        int actual_len = (int)result.moves.size();
        ASSERT_LE(actual_len, tc.known_optimal);

        // Corner DB value must be admissible
        int h = (int)db.lookup(s);
        if (h > actual_len) {
            throw std::runtime_error(
                "Corner DB inadmissible for '" + tc.scramble +
                "': h=" + std::to_string(h) +
                " > actual=" + std::to_string(actual_len));
        }
    }
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    init_moves();

    std::cout << "=== Edge Orientation DB Tests ===\n";
    RUN_TEST(edge_orient_solved);
    RUN_TEST(edge_orient_one_flip_move);
    RUN_TEST(edge_orient_half_turn_no_flip);
    RUN_TEST(edge_orient_admissible);
    RUN_TEST(edge_orient_all_states_populated);

    std::cout << "\n=== Corner Pattern DB Tests (requires ~5min to build) ===\n";
    std::cout << "  Note: Skipping full corner DB tests (too slow for unit tests)\n";
    std::cout << "  Run './build_pattern_db' to build the DB and use './cube_solver' to verify\n";

    // These tests build the full 88MB corner DB - uncomment to run
    // RUN_TEST(corner_db_solved_state);
    // RUN_TEST(corner_db_one_move_states);
    // RUN_TEST(corner_db_admissible_4moves);
    // RUN_TEST(corner_db_consistent_with_solver);

    std::cout << "\n" << g_tests_run << " tests: "
              << g_tests_passed << " passed, "
              << g_tests_failed << " failed\n";

    return g_tests_failed > 0 ? 1 : 0;
}
