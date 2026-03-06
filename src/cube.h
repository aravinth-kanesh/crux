#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <ostream>

// 3×3 cube state: permutation and orientation of 8 corners and 12 edges.
//
// Corner positions:  URF=0, UFL=1, ULB=2, UBR=3, DFR=4, DLF=5, DBL=6, DRB=7
// Edge positions:    UR=0,  UF=1,  UL=2,  UB=3,  DR=4,  DF=5,  DL=6,  DB=7,
//                   FR=8,  FL=9,  BL=10, BR=11
//
// U=Up, D=Down, F=Front, B=Back, L=Left, R=Right

static constexpr int NUM_CORNERS = 8;
static constexpr int NUM_EDGES   = 12;

enum Corner : uint8_t {
    URF = 0, UFL = 1, ULB = 2, UBR = 3,
    DFR = 4, DLF = 5, DBL = 6, DRB = 7
};

enum Edge : uint8_t {
    UR = 0, UF = 1, UL = 2, UB = 3,
    DR = 4, DF = 5, DL = 6, DB = 7,
    FR = 8, FL = 9, BL = 10, BR = 11
};

struct CubeState {
    std::array<uint8_t, NUM_CORNERS> cp;  // cp[i] = cubie at position i
    std::array<uint8_t, NUM_CORNERS> co;  // co[i] = orientation at position i (0,1,2)
    std::array<uint8_t, NUM_EDGES>   ep;  // ep[i] = edge cubie at position i
    std::array<uint8_t, NUM_EDGES>   eo;  // eo[i] = edge orientation at position i (0,1)

    CubeState();  // solved state

    CubeState(
        std::array<uint8_t, NUM_CORNERS> cp,
        std::array<uint8_t, NUM_EDGES>   ep,
        std::array<uint8_t, NUM_CORNERS> co,
        std::array<uint8_t, NUM_EDGES>   eo
    );

    bool operator==(const CubeState& other) const;
    bool operator!=(const CubeState& other) const { return !(*this == other); }

    bool is_solved() const;
    void print() const;
    std::string to_string() const;
};

// The identity (solved) cube
extern const CubeState SOLVED_CUBE;

// compose(a, b): apply b first, then a.  result = a ∘ b
CubeState compose(const CubeState& a, const CubeState& b);

CubeState inverse(const CubeState& c);

// Lehmer-code encoding for pattern database indexing
uint32_t encode_corner_perm  (const std::array<uint8_t, NUM_CORNERS>& cp);
uint32_t encode_corner_orient(const std::array<uint8_t, NUM_CORNERS>& co);
uint32_t encode_edge_orient  (const std::array<uint8_t, NUM_EDGES>&   eo);
uint32_t encode_edge_perm    (const std::array<uint8_t, NUM_EDGES>&   ep);

std::array<uint8_t, NUM_CORNERS> decode_corner_perm  (uint32_t idx);
std::array<uint8_t, NUM_CORNERS> decode_corner_orient(uint32_t idx);
std::array<uint8_t, NUM_EDGES>   decode_edge_orient  (uint32_t idx);

// Corner DB index: perm * 2187 + orient, 8! × 3^7 = 88,179,840 entries
inline uint32_t corner_db_index(const CubeState& c) {
    return encode_corner_perm(c.cp) * 2187u + encode_corner_orient(c.co);
}

// Edge orientation index, 2^11 = 2048 entries
inline uint32_t edge_orient_index(const CubeState& c) {
    return encode_edge_orient(c.eo);
}

// Partial edge DB index: position and orientation of 6 specific edges.
// group 0: edges 0-5  (UR,UF,UL,UB,DR,DF)
// group 1: edges 6-11 (DL,DB,FR,FL,BL,BR)
// group 2: edges 0,2,4,6,8,10 (UR,UL,DR,DL,FR,BL - alternating)
// Uses partial Lehmer rank for positions. Range: [0, P(12,6)*2^6 - 1] = [0, 42,577,919].
uint32_t encode_edge_partial(const uint8_t ep[12], const uint8_t eo[12], int group);
