#include "cube.h"
#include "moves.h"
#include "pattern_db.h"
#include <iostream>
#include <string>
#include <filesystem>
#include <cstring>

int main(int argc, char* argv[]) {
    std::string data_dir = "data";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " [--data-dir DIR]\n";
            return 1;
        }
    }

    std::filesystem::create_directories(data_dir);

    std::cerr << "Initialising move tables...\n";
    init_moves();

    PatternDatabases dbs;

    std::string corner_path    = data_dir + "/corner_pattern.db";
    std::string edge_path      = data_dir + "/edge_orient.db";

    // Build and save corner DB
    std::cerr << "\n=== Building Corner Pattern Database ===\n";
    dbs.corner_db.build();
    if (dbs.corner_db.save(corner_path)) {
        std::cerr << "Saved to " << corner_path << "\n";
    } else {
        std::cerr << "Failed to save corner DB!\n";
        return 1;
    }

    // Build and save edge orientation DB
    std::cerr << "\n=== Building Edge Orientation Database ===\n";
    dbs.edge_orient_db.build();
    if (dbs.edge_orient_db.save(edge_path)) {
        std::cerr << "Saved to " << edge_path << "\n";
    } else {
        std::cerr << "Failed to save edge orient DB!\n";
        return 1;
    }

    std::cerr << "\nAll databases built successfully.\n";

    // Verify a few entries
    std::cerr << "\nVerification:\n";
    std::cerr << "  Solved state corner distance: "
              << (int)dbs.corner_db.lookup(SOLVED_CUBE) << " (expected 0)\n";
    std::cerr << "  Solved state edge orient:     "
              << (int)dbs.edge_orient_db.lookup(SOLVED_CUBE) << " (expected 0)\n";

    auto moves = parse_move_sequence("R U R' U'");
    auto state = apply_moves(SOLVED_CUBE, moves);
    std::cerr << "  After 'R U R' U'': corner=" << (int)dbs.corner_db.lookup(state)
              << " edge_orient=" << (int)dbs.edge_orient_db.lookup(state) << "\n";

    return 0;
}
