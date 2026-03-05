#pragma once
#include "cube.h"
#include "moves.h"
#include <vector>
#include <string>
#include <cstdint>

// ============================================================
// Pattern Databases for IDA* Heuristic
// ============================================================
//
// A pattern database stores the exact number of moves required
// to solve a subset of the cube (ignoring other pieces).
// This gives an admissible (never over-estimating) heuristic.
//
// Corner Pattern Database:
//   - Tracks all 8 corner positions AND orientations
//   - Ignores all 12 edges
//   - Size: 8! * 3^7 = 40320 * 2187 = 88,179,840 entries
//   - Each entry: 1 byte (max distance ≤ 11)
//   - Built via BFS from solved state
//   - Memory: 88 MB
//
// Edge Orientation Database:
//   - Tracks only the orientation of all 12 edges (ignores position)
//   - Size: 2^11 = 2048 entries
//   - Built via BFS; max depth = 11
//   - Memory: 2 KB
// ============================================================

// Corner pattern database (88 MB)
// Maps corner config (perm + orient) -> minimum moves to solve corners
class CornerPatternDB {
public:
    static constexpr uint32_t CORNER_DB_SIZE = 40320u * 2187u; // 88,179,840

    CornerPatternDB();

    // Build the database via BFS from solved state
    // Prints progress to stderr. Can take 5-10 minutes.
    void build();

    // Save to binary file
    bool save(const std::string& path) const;

    // Load from binary file
    bool load(const std::string& path);

    // Is the database fully populated?
    bool is_ready() const { return ready_; }

    // Look up the stored heuristic value for a cube state
    uint8_t lookup(const CubeState& state) const;

    // Direct index lookup — unpacks nibble from packed storage
    uint8_t lookup_idx(uint32_t idx) const {
        uint8_t byte = data_[idx >> 1];
        return (idx & 1) ? (byte >> 4) : (byte & 0x0F);
    }

    // Return number of entries populated (for verification)
    uint32_t populated_count() const;

private:
    // Storage: packed 4-bit nibbles (saves 50% memory vs bytes)
    // data_[i] stores nibbles for entries 2*i and 2*i+1
    std::vector<uint8_t> data_;
    bool ready_ = false;

    // Get/set distance for corner DB index
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
