#include "utils.h"
#include "solver.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>

bool validate_cube(const CubeState& state, std::string& error_msg) {
    {
        std::array<int, NUM_CORNERS> seen = {};
        for (int i = 0; i < NUM_CORNERS; i++) {
            if (state.cp[i] >= NUM_CORNERS) {
                error_msg = "Corner position out of range at slot " + std::to_string(i);
                return false;
            }
            seen[state.cp[i]]++;
        }
        for (int i = 0; i < NUM_CORNERS; i++) {
            if (seen[i] != 1) {
                error_msg = "Corner permutation is not a valid permutation";
                return false;
            }
        }
    }

    for (int i = 0; i < NUM_CORNERS; i++) {
        if (state.co[i] > 2) {
            error_msg = "Corner orientation out of range at slot " + std::to_string(i);
            return false;
        }
    }
    {
        int sum = 0;
        for (int i = 0; i < NUM_CORNERS; i++) sum += state.co[i];
        if (sum % 3 != 0) {
            error_msg = "Corner orientation parity violated (sum=" + std::to_string(sum) + ")";
            return false;
        }
    }

    {
        std::array<int, NUM_EDGES> seen = {};
        for (int i = 0; i < NUM_EDGES; i++) {
            if (state.ep[i] >= NUM_EDGES) {
                error_msg = "Edge position out of range at slot " + std::to_string(i);
                return false;
            }
            seen[state.ep[i]]++;
        }
        for (int i = 0; i < NUM_EDGES; i++) {
            if (seen[i] != 1) {
                error_msg = "Edge permutation is not a valid permutation";
                return false;
            }
        }
    }

    for (int i = 0; i < NUM_EDGES; i++) {
        if (state.eo[i] > 1) {
            error_msg = "Edge orientation out of range at slot " + std::to_string(i);
            return false;
        }
    }
    {
        int sum = 0;
        for (int i = 0; i < NUM_EDGES; i++) sum += state.eo[i];
        if (sum % 2 != 0) {
            error_msg = "Edge orientation parity violated (sum=" + std::to_string(sum) + ")";
            return false;
        }
    }

    auto perm_parity = [](const uint8_t* perm, int n) -> int {
        std::vector<bool> visited(n, false);
        int parity = 0;
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int cycle_len = 0;
                int j = i;
                while (!visited[j]) {
                    visited[j] = true;
                    j = perm[j];
                    cycle_len++;
                }
                parity += (cycle_len - 1) % 2;
            }
        }
        return parity % 2;
    };

    int cp_parity = perm_parity(state.cp.data(), NUM_CORNERS);
    int ep_parity = perm_parity(state.ep.data(), NUM_EDGES);
    if ((cp_parity + ep_parity) % 2 != 0) {
        error_msg = "Permutation parity violated: cube is not solvable";
        return false;
    }

    error_msg = "";
    return true;
}

// ---- Sticker extraction ----
//
// Face indices: U=0 D=1 F=2 B=3 L=4 R=5
//
// Corner face slots (in name order, matching cubie colours):
//   URF=[U,R,F]  UFL=[U,F,L]  ULB=[U,L,B]  UBR=[U,B,R]
//   DFR=[D,F,R]  DLF=[D,L,F]  DBL=[D,B,L]  DRB=[D,R,B]
//
// For corner at position p with cubie c and orientation t,
// face slot j shows colour CORNER_CUBIE_COLOURS[c][(j+t)%3].
// For edge at position p with cubie c and flip f,
// face slot j shows colour EDGE_CUBIE_COLOURS[c][(j+f)%2].

static const uint8_t CORNER_CUBIE_COLOURS[8][3] = {
    {0,5,2}, // URF: U,R,F
    {0,2,4}, // UFL: U,F,L
    {0,4,3}, // ULB: U,L,B
    {0,3,5}, // UBR: U,B,R
    {1,2,5}, // DFR: D,F,R
    {1,4,2}, // DLF: D,L,F
    {1,3,4}, // DBL: D,B,L
    {1,5,3}, // DRB: D,R,B
};

static const uint8_t EDGE_CUBIE_COLOURS[12][2] = {
    {0,5}, // UR
    {0,2}, // UF
    {0,4}, // UL
    {0,3}, // UB
    {1,5}, // DR
    {1,2}, // DF
    {1,4}, // DL
    {1,3}, // DB
    {2,5}, // FR
    {2,4}, // FL
    {3,4}, // BL
    {3,5}, // BR
};

struct StickerRef { uint8_t type; uint8_t piece; uint8_t slot; };
static constexpr StickerRef C(uint8_t p, uint8_t s) { return {1,p,s}; }
static constexpr StickerRef E(uint8_t p, uint8_t s) { return {2,p,s}; }
static constexpr StickerRef CTR()                   { return {0,0,0}; }

// [face][row][col] — see coordinate comments below
// U (face 0): viewed from above, F at bottom → row 0 = back edge, row 2 = front edge
// D (face 1): F at top → row 0 = front edge (adjacent to F face bottom)
// F (face 2): viewed from front, U at top, L on left
// B (face 3): viewed from back, U at top, col 0 = R-side-from-back (our right)
// L (face 4): viewed from left, U at top, col 0 = B-side, col 2 = F-side
// R (face 5): viewed from right, U at top, col 0 = F-side, col 2 = B-side
static const StickerRef FACE_STICKERS[6][3][3] = {
    // U
    {{ C(2,0), E(3,0), C(3,0) },
     { E(2,0), CTR(),  E(0,0) },
     { C(1,0), E(1,0), C(0,0) }},
    // D
    {{ C(5,0), E(5,0), C(4,0) },
     { E(6,0), CTR(),  E(4,0) },
     { C(6,0), E(7,0), C(7,0) }},
    // F
    {{ C(1,1), E(1,1), C(0,2) },
     { E(9,0), CTR(),  E(8,0) },
     { C(5,2), E(5,1), C(4,1) }},
    // B
    {{ C(3,1), E(3,1), C(2,2) },
     { E(11,0),CTR(),  E(10,0)},
     { C(7,2), E(7,1), C(6,1) }},
    // L
    {{ C(2,1), E(2,1), C(1,2) },
     { E(10,1),CTR(),  E(9,1) },
     { C(6,2), E(6,1), C(5,1) }},
    // R
    {{ C(0,1), E(0,1), C(3,2) },
     { E(8,1), CTR(),  E(11,1)},
     { C(4,2), E(4,1), C(7,1) }},
};

static uint8_t get_sticker(const CubeState& s, int face, int row, int col) {
    const StickerRef& ref = FACE_STICKERS[face][row][col];
    if (ref.type == 0) return (uint8_t)face;
    if (ref.type == 1) {
        uint8_t c = s.cp[ref.piece], t = s.co[ref.piece];
        return CORNER_CUBIE_COLOURS[c][(ref.slot + t) % 3];
    }
    uint8_t e = s.ep[ref.piece], f = s.eo[ref.piece];
    return EDGE_CUBIE_COLOURS[e][(ref.slot + f) % 2];
}

// ---- ANSI cross-layout printer ----
//
// Each sticker = 2 spaces with an ANSI background colour.
// Falls back to letter codes (W/Y/G/B/O/R) if NO_COLOR env var is set.
// Layout (each face 6 chars wide, U/D indented by 6):
//       [U]
//  [L][F][R][B]
//       [D]

static const char* ANSI_BG[6] = {
    "\033[47m",        // U = White
    "\033[43m",        // D = Yellow
    "\033[42m",        // F = Green
    "\033[44m",        // B = Blue
    "\033[48;5;202m",  // L = Orange (256-colour)
    "\033[41m",        // R = Red
};
static const char* ANSI_RESET = "\033[0m";
static const char  COLOUR_LETTERS[6] = { 'W', 'Y', 'G', 'B', 'O', 'R' };

void print_cube_ascii(const CubeState& state) {
    const bool no_color = (std::getenv("NO_COLOR") != nullptr);

    auto cell = [&](int face, int row, int col) -> std::string {
        uint8_t c = get_sticker(state, face, row, col);
        if (no_color) return std::string(1, COLOUR_LETTERS[c]) + ' ';
        return std::string(ANSI_BG[c]) + "  " + ANSI_RESET;
    };

    const std::string gap(6, ' ');

    for (int r = 0; r < 3; r++) {
        std::cout << gap;
        for (int c = 0; c < 3; c++) std::cout << cell(0, r, c);
        std::cout << '\n';
    }
    for (int r = 0; r < 3; r++) {
        for (int f : {4, 2, 5, 3})
            for (int c = 0; c < 3; c++) std::cout << cell(f, r, c);
        std::cout << '\n';
    }
    for (int r = 0; r < 3; r++) {
        std::cout << gap;
        for (int c = 0; c < 3; c++) std::cout << cell(1, r, c);
        std::cout << '\n';
    }
}

std::string format_solve_stats(const SolveResult& result) {
    std::ostringstream oss;
    if (result.found) {
        oss << "Solution found: " << format_move_sequence(result.moves)
            << "\nMoves: " << result.moves.size()
            << "\nNodes explored: " << result.nodes_explored
            << "\nTime: " << result.time_seconds << "s";
    } else {
        oss << "No solution found (max depth reached)\n"
            << "Nodes explored: " << result.nodes_explored
            << "\nTime: " << result.time_seconds << "s";
    }
    return oss.str();
}
