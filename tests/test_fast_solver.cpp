// Tests for FastIDASolver and the three coordinate move tables.
// No corner pattern DB required; table tests are standalone.
#include "../src/cube.h"
#include "../src/moves.h"
#include "../src/solver.h"
#include "../src/fast_solver.h"
#include "../src/pattern_db.h"
#include <iostream>
#include <cassert>
#include <sstream>

static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    g_tests_run++; \
    std::cout << "  " #name "... "; \
    try { test_##name(); std::cout << "PASS\n"; g_tests_passed++; } \
    catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << "\n"; g_tests_failed++; } \
    catch (...) { std::cout << "FAIL\n"; g_tests_failed++; } \
} while(0)

#define ASSERT(cond) do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond " at line " + std::to_string(__LINE__)); } while(0)
#define ASSERT_EQ(a,b) do { if ((a)!=(b)) { std::ostringstream ss; ss << (a) << " != " << (b) << " at line " << __LINE__; throw std::runtime_error(ss.str()); } } while(0)

// CP_MOVE_TABLE tests

TEST(cp_move_table_from_solved) {
    // From the solved state cp_idx=0, right and left multiplication agree,
    // so CP_MOVE_TABLE[m][0] must match apply_move(SOLVED, m).
    for (int m = 0; m < NUM_MOVES; m++) {
        CubeState after = apply_move(SOLVED_CUBE, m);
        uint32_t expected = encode_corner_perm(after.cp);
        uint32_t got      = CP_MOVE_TABLE[m][0];
        if (got != expected)
            throw std::runtime_error(
                "CP_MOVE_TABLE wrong for " + MOVE_NAMES[m] +
                ": got " + std::to_string(got) +
                " expected " + std::to_string(expected));
    }
}

TEST(cp_move_table_chained) {
    // Apply R then U via table; compare against right-multiplication result.
    // (Right mult: compose(state, mv); uses the existing compose() function.)
    CubeState after_R  = compose(SOLVED_CUBE, MOVE_TABLE[MoveIndex::R]);
    CubeState after_RU = compose(after_R,     MOVE_TABLE[MoveIndex::U]);
    uint32_t expected = encode_corner_perm(after_RU.cp);

    uint32_t via_table = CP_MOVE_TABLE[MoveIndex::U][CP_MOVE_TABLE[MoveIndex::R][0]];
    ASSERT_EQ(via_table, expected);
}

TEST(cp_move_table_inverse) {
    // Applying a move then its inverse returns cp_idx to 0.
    for (int m = 0; m < NUM_MOVES; m++) {
        uint32_t after = CP_MOVE_TABLE[m][0];
        uint32_t back  = CP_MOVE_TABLE[INVERSE_MOVE[m]][after];
        ASSERT_EQ(back, 0u);
    }
}

// CO_MOVE_TABLE tests

TEST(co_move_table_from_solved) {
    // CO_MOVE_TABLE[m][0] must match the co_idx after right-multiplying
    // the solved state by each move.
    for (int m = 0; m < NUM_MOVES; m++) {
        CubeState after = compose(SOLVED_CUBE, MOVE_TABLE[m]);
        uint32_t expected = encode_corner_orient(after.co);
        uint32_t got      = CO_MOVE_TABLE[m][0];
        if (got != expected)
            throw std::runtime_error(
                "CO_MOVE_TABLE wrong for " + MOVE_NAMES[m] +
                ": got " + std::to_string(got) +
                " expected " + std::to_string(expected));
    }
}

TEST(co_move_table_inverse) {
    // Applying a move then its inverse returns co_idx to 0.
    for (int m = 0; m < NUM_MOVES; m++) {
        uint32_t after = CO_MOVE_TABLE[m][0];
        uint32_t back  = CO_MOVE_TABLE[INVERSE_MOVE[m]][after];
        ASSERT_EQ(back, 0u);
    }
}

// EO_MOVE_TABLE tests

TEST(eo_move_table_from_solved) {
    // EO_MOVE_TABLE[m][0] must match the eo_idx after right-multiplying
    // the solved state by each move.
    for (int m = 0; m < NUM_MOVES; m++) {
        CubeState after = compose(SOLVED_CUBE, MOVE_TABLE[m]);
        uint32_t expected = encode_edge_orient(after.eo);
        uint32_t got      = EO_MOVE_TABLE[m][0];
        if (got != expected)
            throw std::runtime_error(
                "EO_MOVE_TABLE wrong for " + MOVE_NAMES[m] +
                ": got " + std::to_string(got) +
                " expected " + std::to_string(expected));
    }
}

TEST(eo_move_table_inverse) {
    // Applying a move then its inverse returns eo_idx to 0.
    for (int m = 0; m < NUM_MOVES; m++) {
        uint32_t after = EO_MOVE_TABLE[m][0];
        uint32_t back  = EO_MOVE_TABLE[INVERSE_MOVE[m]][after];
        ASSERT_EQ(back, 0u);
    }
}

// Cross-consistency: table vs direct encode after compose()

TEST(co_move_table_chained) {
    // Apply F then B via CO table; compare against right-multiplication result.
    CubeState after_F  = compose(SOLVED_CUBE, MOVE_TABLE[MoveIndex::F]);
    CubeState after_FB = compose(after_F,     MOVE_TABLE[MoveIndex::B]);
    uint32_t expected = encode_corner_orient(after_FB.co);

    uint32_t via_table = CO_MOVE_TABLE[MoveIndex::B][CO_MOVE_TABLE[MoveIndex::F][0]];
    ASSERT_EQ(via_table, expected);
}

TEST(eo_move_table_chained) {
    // Apply F then B via EO table; F and B both flip edges, so a good test.
    CubeState after_F  = compose(SOLVED_CUBE, MOVE_TABLE[MoveIndex::F]);
    CubeState after_FB = compose(after_F,     MOVE_TABLE[MoveIndex::B]);
    uint32_t expected = encode_edge_orient(after_FB.eo);

    uint32_t via_table = EO_MOVE_TABLE[MoveIndex::B][EO_MOVE_TABLE[MoveIndex::F][0]];
    ASSERT_EQ(via_table, expected);
}

int main() {
    init_moves();
    build_cp_move_table();
    build_co_move_table();
    build_eo_move_table();

    std::cout << "=== Fast Solver Tests ===\n";
    RUN_TEST(cp_move_table_from_solved);
    RUN_TEST(cp_move_table_chained);
    RUN_TEST(cp_move_table_inverse);
    RUN_TEST(co_move_table_from_solved);
    RUN_TEST(co_move_table_inverse);
    RUN_TEST(eo_move_table_from_solved);
    RUN_TEST(eo_move_table_inverse);
    RUN_TEST(co_move_table_chained);
    RUN_TEST(eo_move_table_chained);

    std::cout << "\n" << g_tests_run << " tests: "
              << g_tests_passed << " passed, "
              << g_tests_failed << " failed\n";

    return g_tests_failed > 0 ? 1 : 0;
}
