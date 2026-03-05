#pragma once
#include "cube.h"
#include "moves.h"
#include <vector>
#include <string>
#include <cstdint>

// Pattern databases for IDA* heuristic lookup.
// CornerPatternDB: all 8 corners (perm + orient), ~42 MB nibble-packed, built via BFS.
// EdgeOrientDB: all 12 edge orientations, 2 KB.

// Corner pattern database (~42 MB nibble-packed)
// Maps corner config (perm + orient) -> minimum moves to solve corners
class CornerPatternDB {
public:
    static constexpr uint32_t CORNER_DB_SIZE = 40320u * 2187u; // 88,179,840

    CornerPatternDB();

    void build();  // BFS from solved; prints progress to stderr, ~5-10 min
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    bool is_ready() const { return ready_; }
    uint8_t lookup(const CubeState& state) const;

    // Direct index lookup; unpacks nibble from packed storage
    uint8_t lookup_idx(uint32_t idx) const {
        uint8_t byte = data_[idx >> 1];
        return (idx & 1) ? (byte >> 4) : (byte & 0x0F);
    }

    uint32_t populated_count() const;

private:
    // Packed nibbles: data_[i] holds entries 2i (low) and 2i+1 (high)
    std::vector<uint8_t> data_;
    bool ready_ = false;

    uint8_t get(uint32_t idx) const;
    void set(uint32_t idx, uint8_t val);
};

// Edge orientation database (2 KB)
// Maps edge orientation config -> minimum moves to solve edge orientations
class EdgeOrientDB {
public:
    static constexpr uint32_t EDGE_ORIENT_DB_SIZE = 2048u; // 2^11

    EdgeOrientDB();

    void build();

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    bool is_ready() const { return ready_; }

    uint8_t lookup(const CubeState& state) const;
    uint8_t lookup_idx(uint32_t idx) const { return data_[idx]; }

private:
    std::array<uint8_t, EDGE_ORIENT_DB_SIZE> data_;
    bool ready_ = false;
};

// Combined databases (used by the main heuristic)
struct PatternDatabases {
    CornerPatternDB corner_db;
    EdgeOrientDB edge_orient_db;

    // Try to load from directory. Build if not found.
    // Returns true if loaded from disk, false if built fresh.
    bool load_or_build(const std::string& data_dir);

    // Combined heuristic value: max of all DBs
    int heuristic(const CubeState& state) const;
};
