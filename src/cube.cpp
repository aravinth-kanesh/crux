#include "cube.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>

const CubeState SOLVED_CUBE;

CubeState::CubeState() {
    for (int i = 0; i < NUM_CORNERS; i++) { cp[i] = (uint8_t)i; co[i] = 0; }
    for (int i = 0; i < NUM_EDGES;   i++) { ep[i] = (uint8_t)i; eo[i] = 0; }
}

CubeState::CubeState(
    std::array<uint8_t, NUM_CORNERS> cp_,
    std::array<uint8_t, NUM_EDGES>   ep_,
    std::array<uint8_t, NUM_CORNERS> co_,
    std::array<uint8_t, NUM_EDGES>   eo_
) : cp(cp_), co(co_), ep(ep_), eo(eo_) {}

bool CubeState::operator==(const CubeState& other) const {
    return cp == other.cp && co == other.co &&
           ep == other.ep && eo == other.eo;
}

bool CubeState::is_solved() const { return *this == SOLVED_CUBE; }

void CubeState::print() const { std::cout << to_string() << '\n'; }

std::string CubeState::to_string() const {
    std::ostringstream oss;
    oss << "Corners pos: ";
    for (int i = 0; i < NUM_CORNERS; i++) oss << (int)cp[i] << ' ';
    oss << "\nCorners ori: ";
    for (int i = 0; i < NUM_CORNERS; i++) oss << (int)co[i] << ' ';
    oss << "\nEdges   pos: ";
    for (int i = 0; i < NUM_EDGES;   i++) oss << (int)ep[i] << ' ';
    oss << "\nEdges   ori: ";
    for (int i = 0; i < NUM_EDGES;   i++) oss << (int)eo[i] << ' ';
    return oss.str();
}

// compose(a, b): apply b first, then a.
// result.cp[i] = a.cp[b.cp[i]],  result.co[i] = (a.co[b.cp[i]] + b.co[i]) % 3
CubeState compose(const CubeState& a, const CubeState& b) {
    CubeState res;
    for (int i = 0; i < NUM_CORNERS; i++) {
        res.cp[i] = a.cp[b.cp[i]];
        res.co[i] = (a.co[b.cp[i]] + b.co[i]) % 3;
    }
    for (int i = 0; i < NUM_EDGES; i++) {
        res.ep[i] = a.ep[b.ep[i]];
        res.eo[i] = (a.eo[b.ep[i]] + b.eo[i]) % 2;
    }
    return res;
}

CubeState inverse(const CubeState& c) {
    CubeState inv;
    for (int i = 0; i < NUM_CORNERS; i++) {
        inv.cp[c.cp[i]] = (uint8_t)i;
        inv.co[c.cp[i]] = (3 - c.co[i]) % 3;
    }
    for (int i = 0; i < NUM_EDGES; i++) {
        inv.ep[c.ep[i]] = (uint8_t)i;
        inv.eo[c.ep[i]] = c.eo[i];  // flip is its own inverse
    }
    return inv;
}

// Lehmer (factoriadic) encoding: maps a permutation of {0..n-1} to [0, n!-1].
// O(n²), fine for n ≤ 12.
static uint32_t lehmer_encode(const uint8_t* perm, int n) {
    uint32_t idx = 0;
    for (int i = 0; i < n; i++) {
        uint32_t count = 0;
        for (int j = i + 1; j < n; j++)
            if (perm[j] < perm[i]) count++;
        uint32_t fact = 1;
        for (int k = 1; k <= (n - 1 - i); k++) fact *= k;
        idx += count * fact;
    }
    return idx;
}

uint32_t encode_corner_perm  (const std::array<uint8_t, NUM_CORNERS>& cp) { return lehmer_encode(cp.data(), NUM_CORNERS); }
uint32_t encode_edge_perm    (const std::array<uint8_t, NUM_EDGES>&   ep) { return lehmer_encode(ep.data(), NUM_EDGES); }

uint32_t encode_corner_orient(const std::array<uint8_t, NUM_CORNERS>& co) {
    // 7 digits base-3; the 8th is constrained by sum ≡ 0 mod 3
    uint32_t idx = 0;
    for (int i = 0; i < NUM_CORNERS - 1; i++) idx = idx * 3 + co[i];
    return idx;
}

uint32_t encode_edge_orient(const std::array<uint8_t, NUM_EDGES>& eo) {
    // 11 digits base-2; the 12th is constrained by sum ≡ 0 mod 2
    uint32_t idx = 0;
    for (int i = 0; i < NUM_EDGES - 1; i++) idx = idx * 2 + eo[i];
    return idx;
}

std::array<uint8_t, NUM_CORNERS> decode_corner_perm(uint32_t idx) {
    std::array<uint8_t, NUM_CORNERS> perm;
    std::vector<uint8_t> available = {0,1,2,3,4,5,6,7};
    static const uint32_t fact[NUM_CORNERS] = {5040,720,120,24,6,2,1,1};
    for (int i = 0; i < NUM_CORNERS; i++) {
        uint32_t k = idx / fact[i];
        idx %= fact[i];
        perm[i] = available[k];
        available.erase(available.begin() + k);
    }
    return perm;
}

std::array<uint8_t, NUM_CORNERS> decode_corner_orient(uint32_t idx) {
    std::array<uint8_t, NUM_CORNERS> co;
    uint32_t sum = 0;
    for (int i = NUM_CORNERS - 2; i >= 0; i--) {
        co[i] = idx % 3;
        sum += co[i];
        idx /= 3;
    }
    co[NUM_CORNERS - 1] = (3 - sum % 3) % 3;
    return co;
}

std::array<uint8_t, NUM_EDGES> decode_edge_orient(uint32_t idx) {
    std::array<uint8_t, NUM_EDGES> eo;
    uint32_t sum = 0;
    for (int i = NUM_EDGES - 2; i >= 0; i--) {
        eo[i] = idx % 2;
        sum += eo[i];
        idx /= 2;
    }
    eo[NUM_EDGES - 1] = sum % 2;
    return eo;
}
