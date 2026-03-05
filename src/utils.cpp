#include "utils.h"
#include "solver.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>

bool validate_cube(const CubeState& state, std::string& error_msg) {
    // Check corner permutation is a valid permutation of {0..7}
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

    // Check corner orientations
    for (int i = 0; i < NUM_CORNERS; i++) {
        if (state.co[i] > 2) {
            error_msg = "Corner orientation out of range at slot " + std::to_string(i);
            return false;
        }
    }
    // Sum of corner orientations must be 0 mod 3
    {
        int sum = 0;
        for (int i = 0; i < NUM_CORNERS; i++) sum += state.co[i];
        if (sum % 3 != 0) {
            error_msg = "Corner orientation parity violated (sum=" + std::to_string(sum) + ")";
            return false;
        }
    }

    // Check edge permutation
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

    // Check edge orientations
    for (int i = 0; i < NUM_EDGES; i++) {
        if (state.eo[i] > 1) {
            error_msg = "Edge orientation out of range at slot " + std::to_string(i);
            return false;
        }
    }
    // Sum of edge orientations must be 0 mod 2
    {
        int sum = 0;
        for (int i = 0; i < NUM_EDGES; i++) sum += state.eo[i];
        if (sum % 2 != 0) {
            error_msg = "Edge orientation parity violated (sum=" + std::to_string(sum) + ")";
            return false;
        }
    }

    // Check combined permutation parity
    // Corner perm parity XOR edge perm parity must be even
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

// ASCII art visualization of cube
// Uses face colours: U=W(white), D=Y(yellow), F=G(green),
//                    B=B(blue), L=O(orange), R=R(red)
// Layout (cross):
//       UUU
//       UUU
//       UUU
//  LLLFFFRRRBBB
//  LLLFFFRRRBBB
//  LLLFFFRRRBBB
//       DDD
//       DDD
//       DDD
void print_cube_ascii(const CubeState& state) {
    std::cout << "Corner positions: ";
    for (int i = 0; i < NUM_CORNERS; i++) std::cout << (int)state.cp[i] << " ";
    std::cout << "\nCorner orienta: ";
    for (int i = 0; i < NUM_CORNERS; i++) std::cout << (int)state.co[i] << " ";
    std::cout << "\nEdge positions: ";
    for (int i = 0; i < NUM_EDGES; i++) std::cout << (int)state.ep[i] << " ";
    std::cout << "\nEdge orientati: ";
    for (int i = 0; i < NUM_EDGES; i++) std::cout << (int)state.eo[i] << " ";
    std::cout << "\n";
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
