#include "pattern_db.h"
#include <queue>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstring>
#include <chrono>

// ============================================================
// CornerPatternDB
// ============================================================

CornerPatternDB::CornerPatternDB() {
    // Packed nibbles: CORNER_DB_SIZE entries, 4 bits each
    // Need ceil(CORNER_DB_SIZE / 2) bytes
    data_.assign((CORNER_DB_SIZE + 1) / 2, 0xFF);  // 0xFF = both nibbles unvisited (15)
}

uint8_t CornerPatternDB::get(uint32_t idx) const {
    uint8_t byte = data_[idx / 2];
    return (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
}

void CornerPatternDB::set(uint32_t idx, uint8_t val) {
    uint8_t& byte = data_[idx / 2];
    if (idx % 2 == 0) {
        byte = (byte & 0xF0) | (val & 0x0F);
    } else {
        byte = (byte & 0x0F) | ((val & 0x0F) << 4);
    }
}

uint8_t CornerPatternDB::lookup(const CubeState& state) const {
    return get(corner_db_index(state));
}

uint32_t CornerPatternDB::populated_count() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < CORNER_DB_SIZE; i++) {
        if (get(i) != 15) count++;
    }
    return count;
}

// ============================================================
// Build corner pattern DB via BFS
//
// State space: 8! * 3^7 = 88,179,840 states
// States indexed by (corner_perm_idx * 2187 + corner_orient_idx)
//
// BFS from solved state. For each state, apply all 18 moves
// and record the distance to the new state.
// ============================================================
void CornerPatternDB::build() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // Reset: all entries unvisited (stored as 15 in nibbles via 0xFF bytes)
    std::fill(data_.begin(), data_.end(), 0xFF);

    // BFS queue stores corner DB indices
    std::queue<uint32_t> q;

    // Solved state
    uint32_t solved_idx = corner_db_index(SOLVED_CUBE);
    set(solved_idx, 0);
    q.push(solved_idx);

    uint64_t visited = 1;
    int current_depth = 0;

    std::cerr << "Building corner pattern database (" << CORNER_DB_SIZE << " states)...\n";

    while (!q.empty()) {
        uint32_t idx = q.front();
        q.pop();

        uint8_t dist = get(idx);
        if (dist != current_depth) {
            current_depth = dist;
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            std::cerr << "  Depth " << current_depth
                      << ": " << visited << " states visited"
                      << " (" << elapsed << "s)\n";
        }

        // Decode the corner state from the DB index
        uint32_t cp_idx = idx / 2187u;
        uint32_t co_idx = idx % 2187u;
        auto cp = decode_corner_perm(cp_idx);
        auto co = decode_corner_orient(co_idx);

        // Apply all 18 moves
        for (int m = 0; m < NUM_MOVES; m++) {
            const CubeState& mv = MOVE_TABLE[m];

            // Apply move to corner permutation
            std::array<uint8_t, NUM_CORNERS> new_cp;
            std::array<uint8_t, NUM_CORNERS> new_co;
            for (int i = 0; i < NUM_CORNERS; i++) {
                new_cp[i] = cp[mv.cp[i]];
                // compose(state, mv): new_co[i] = (state.co[mv.cp[i]] + mv.co[i]) % 3
                new_co[i] = (uint8_t)((co[mv.cp[i]] + mv.co[i]) % 3);
            }

            uint32_t new_cp_idx = encode_corner_perm(new_cp);
            uint32_t new_co_idx = encode_corner_orient(new_co);
            uint32_t new_idx = new_cp_idx * 2187u + new_co_idx;

            if (get(new_idx) == 15) {  // unvisited
                set(new_idx, dist + 1);
                q.push(new_idx);
                visited++;
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cerr << "Corner DB complete: " << visited << " states in "
              << elapsed << "s\n";

    ready_ = true;
}

bool CornerPatternDB::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return false;
    }
    f.write(reinterpret_cast<const char*>(data_.data()), (long)data_.size());
    return f.good();
}

bool CornerPatternDB::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    // Check file size
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    size_t expected = (CORNER_DB_SIZE + 1) / 2;
    if (file_size != expected) {
        std::cerr << "Corner DB file size mismatch: expected " << expected
                  << " bytes, got " << file_size << "\n";
        return false;
    }

    f.read(reinterpret_cast<char*>(data_.data()), (long)data_.size());
    if (!f) return false;

    ready_ = true;
    return true;
}

// ============================================================
// EdgeOrientDB
// ============================================================

EdgeOrientDB::EdgeOrientDB() {
    data_.fill(255);  // 255 = unvisited
}

uint8_t EdgeOrientDB::lookup(const CubeState& state) const {
    return data_[edge_orient_index(state)];
}

void EdgeOrientDB::build() {
    data_.fill(255);

    std::queue<uint32_t> q;
    uint32_t solved_idx = edge_orient_index(SOLVED_CUBE);
    data_[solved_idx] = 0;
    q.push(solved_idx);

    uint64_t visited = 1;
    std::cerr << "Building edge orientation database (" << EDGE_ORIENT_DB_SIZE << " states)...\n";

    while (!q.empty()) {
        uint32_t idx = q.front();
        q.pop();

        uint8_t dist = data_[idx];
        auto eo = decode_edge_orient(idx);

        for (int m = 0; m < NUM_MOVES; m++) {
            const CubeState& mv = MOVE_TABLE[m];

            std::array<uint8_t, NUM_EDGES> new_eo;
            for (int i = 0; i < NUM_EDGES; i++) {
                // compose(state, mv): new_eo[i] = (eo[mv.ep[i]] + mv.eo[i]) % 2
                // mv.ep[i] is the source position of position i after the move
                new_eo[i] = (uint8_t)((eo[mv.ep[i]] + mv.eo[i]) % 2);
            }

            uint32_t new_idx = encode_edge_orient(new_eo);
            if (data_[new_idx] == 255) {
                data_[new_idx] = dist + 1;
                q.push(new_idx);
                visited++;
            }
        }
    }

    std::cerr << "Edge orient DB complete: " << visited << " states\n";
    ready_ = true;
}

bool EdgeOrientDB::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data_.data()), (long)data_.size());
    return f.good();
}

bool EdgeOrientDB::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(data_.data()), (long)data_.size());
    if (!f) return false;
    ready_ = true;
    return true;
}

// ============================================================
// PatternDatabases combined
// ============================================================
bool PatternDatabases::load_or_build(const std::string& data_dir) {
    std::string corner_path    = data_dir + "/corner_pattern.db";
    std::string edge_orient_path = data_dir + "/edge_orient.db";

    bool loaded_from_disk = true;

    // Try to load corner DB
    if (!corner_db.load(corner_path)) {
        std::cerr << "Corner DB not found, building...\n";
        corner_db.build();
        if (!corner_db.save(corner_path)) {
            std::cerr << "Warning: could not save corner DB to " << corner_path << "\n";
        }
        loaded_from_disk = false;
    } else {
        std::cerr << "Loaded corner pattern DB from " << corner_path << "\n";
    }

    // Try to load edge orientation DB
    if (!edge_orient_db.load(edge_orient_path)) {
        std::cerr << "Edge orient DB not found, building...\n";
        edge_orient_db.build();
        if (!edge_orient_db.save(edge_orient_path)) {
            std::cerr << "Warning: could not save edge orient DB to " << edge_orient_path << "\n";
        }
        loaded_from_disk = false;
    } else {
        std::cerr << "Loaded edge orientation DB from " << edge_orient_path << "\n";
    }

    return loaded_from_disk;
}

int PatternDatabases::heuristic(const CubeState& state) const {
    int h = 0;
    if (corner_db.is_ready()) {
        h = std::max(h, (int)corner_db.lookup(state));
    }
    if (edge_orient_db.is_ready()) {
        h = std::max(h, (int)edge_orient_db.lookup(state));
    }
    return h;
}
