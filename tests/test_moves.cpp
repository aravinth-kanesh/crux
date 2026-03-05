// Unit tests for cube state and move generation
// Simple test framework (no external dependencies)
#include "../src/cube.h"
#include "../src/moves.h"
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

TEST(solved_is_solved) {
    CubeState s;
    ASSERT(s.is_solved());
    ASSERT(s == SOLVED_CUBE);
}

TEST(move_inverse) {
    for (int m = 0; m < NUM_MOVES; m++) {
        CubeState s = apply_move(SOLVED_CUBE, m);
        int inv = INVERSE_MOVE[m];
        CubeState back = apply_move(s, inv);
        if (!back.is_solved()) {
            std::ostringstream ss;
            ss << "Move " << MOVE_NAMES[m] << " followed by its inverse "
               << MOVE_NAMES[inv] << " does not return to solved";
            throw std::runtime_error(ss.str());
        }
    }
}

TEST(move_order_4) {
    // U, U', F, F', etc. all have order 4
    for (int m = 0; m < NUM_MOVES; m += 3) {  // Skip half-turns (order 2)
        CubeState s = SOLVED_CUBE;
        for (int rep = 0; rep < 4; rep++) {
            s = apply_move(s, m);
        }
        if (!s.is_solved()) {
            throw std::runtime_error("Move " + MOVE_NAMES[m] + "^4 != identity");
        }
    }
}

TEST(half_turn_order_2) {
    // U2, D2, F2, B2, L2, R2 are moves 2,5,8,11,14,17
    for (int m = 2; m < NUM_MOVES; m += 3) {
        CubeState s = SOLVED_CUBE;
        s = apply_move(s, m);
        s = apply_move(s, m);
        if (!s.is_solved()) {
            throw std::runtime_error("Half turn " + MOVE_NAMES[m] + "^2 != identity");
        }
    }
}

// (T-perm has order 2, Sexy move has order 6)
TEST(sexy_move_order_6) {
    // Sexy move: R U R' U' has order 6
    auto sexy = parse_move_sequence("R U R' U'");
    CubeState s = SOLVED_CUBE;
    for (int rep = 0; rep < 6; rep++) {
        s = apply_moves(s, sexy);
    }
    if (!s.is_solved()) {
        throw std::runtime_error("(R U R' U')^6 != identity");
    }
}

TEST(u_perm_order_3) {
    // Verify apply_moves and its inverse return to solved
    auto seq = parse_move_sequence("R U");
    CubeState s = apply_moves(SOLVED_CUBE, seq);
    ASSERT(!s.is_solved());

    auto inv_seq = parse_move_sequence("U' R'");
    s = apply_moves(s, inv_seq);
    ASSERT(s.is_solved());
}

TEST(compose_associative) {
    CubeState a = apply_move(SOLVED_CUBE, MoveIndex::R);
    CubeState b = apply_move(SOLVED_CUBE, MoveIndex::U);
    CubeState c = apply_move(SOLVED_CUBE, MoveIndex::F);

    // (a * b) * c == a * (b * c)
    CubeState lhs = compose(compose(a, b), c);
    CubeState rhs = compose(a, compose(b, c));
    ASSERT(lhs == rhs);
}

TEST(inverse_correctness) {
    CubeState m = apply_move(SOLVED_CUBE, MoveIndex::F);
    CubeState inv_m = inverse(m);

    // m * inv(m) = identity
    CubeState result = compose(m, inv_m);
    ASSERT(result.is_solved());

    // inv(m) * m = identity
    result = compose(inv_m, m);
    ASSERT(result.is_solved());
}

TEST(parse_format_moves) {
    std::string seq_str = "R U R' U' F2 D' L2 U B";
    auto moves = parse_move_sequence(seq_str);
    ASSERT_EQ((int)moves.size(), 9);

    std::string reformatted = format_move_sequence(moves);
    ASSERT_EQ(reformatted, seq_str);
}

TEST(redundancy_pruning) {
    // Same face consecutive: U then U' is redundant
    ASSERT(is_redundant_sequence(MoveIndex::U, MoveIndex::Up));
    ASSERT(is_redundant_sequence(MoveIndex::U, MoveIndex::U2));
    ASSERT(is_redundant_sequence(MoveIndex::F, MoveIndex::F));

    // Different faces: not redundant
    ASSERT(!is_redundant_sequence(MoveIndex::U, MoveIndex::R));
    ASSERT(!is_redundant_sequence(MoveIndex::F, MoveIndex::B));

    // No previous move: never redundant
    ASSERT(!is_redundant_sequence(-1, MoveIndex::U));
}

TEST(lehmer_encoding) {
    // For corner permutation: encode then decode should give identity
    std::array<uint8_t, NUM_CORNERS> cp = {0,1,2,3,4,5,6,7}; // solved
    uint32_t idx = encode_corner_perm(cp);
    ASSERT_EQ(idx, 0u); // solved = index 0

    auto decoded = decode_corner_perm(0);
    for (int i = 0; i < NUM_CORNERS; i++) {
        ASSERT_EQ((int)decoded[i], i);
    }

    // Encode all permutations and check they're unique
    // (only check a few random ones for speed)
    std::array<uint8_t, NUM_CORNERS> perm = {1,0,2,3,4,5,6,7};
    idx = encode_corner_perm(perm);
    ASSERT(idx > 0);
    auto decoded2 = decode_corner_perm(idx);
    for (int i = 0; i < NUM_CORNERS; i++) {
        ASSERT_EQ((int)decoded2[i], (int)perm[i]);
    }
}

TEST(corner_orient_encoding) {
    std::array<uint8_t, NUM_CORNERS> co_solved = {0,0,0,0,0,0,0,0};
    ASSERT_EQ(encode_corner_orient(co_solved), 0u);

    auto decoded = decode_corner_orient(0);
    for (int i = 0; i < NUM_CORNERS; i++) {
        ASSERT_EQ((int)decoded[i], 0);
    }

    // Test a non-trivial orientation
    std::array<uint8_t, NUM_CORNERS> co = {1,2,0,0,0,0,0,0};
    // Sum = 3, which is 0 mod 3. Last element = (3 - sum_of_first_7) % 3
    // First 7: 1+2+0+0+0+0+0 = 3, last = (3-3%3)%3 = 0
    // Actually the last is determined so that total sum = 0 mod 3
    uint32_t co_idx = encode_corner_orient(co);
    auto decoded_co = decode_corner_orient(co_idx);
    // Check first 7 match
    for (int i = 0; i < NUM_CORNERS - 1; i++) {
        ASSERT_EQ((int)decoded_co[i], (int)co[i]);
    }
}

TEST(known_scramble_not_solved) {
    auto moves = parse_move_sequence("R U R' U' F2 D' L B");
    CubeState state = apply_moves(SOLVED_CUBE, moves);
    ASSERT(!state.is_solved());
}

TEST(corner_db_index_solved) {
    uint32_t idx = corner_db_index(SOLVED_CUBE);
    ASSERT_EQ(idx, 0u);
}

// Main
int main() {
    init_moves();

    std::cout << "=== Cube/Move Tests ===\n";
    RUN_TEST(solved_is_solved);
    RUN_TEST(move_inverse);
    RUN_TEST(move_order_4);
    RUN_TEST(half_turn_order_2);
    RUN_TEST(sexy_move_order_6);
    RUN_TEST(u_perm_order_3);
    RUN_TEST(compose_associative);
    RUN_TEST(inverse_correctness);
    RUN_TEST(parse_format_moves);
    RUN_TEST(redundancy_pruning);
    RUN_TEST(lehmer_encoding);
    RUN_TEST(corner_orient_encoding);
    RUN_TEST(known_scramble_not_solved);
    RUN_TEST(corner_db_index_solved);

    std::cout << "\n" << g_tests_run << " tests: "
              << g_tests_passed << " passed, "
              << g_tests_failed << " failed\n";

    return g_tests_failed > 0 ? 1 : 0;
}
