#pragma once
#include "cube.h"
#include "pattern_db.h"
#include <functional>

// Admissible heuristic wrappers for IDASolver (pattern DB path).

// h = corner_db.lookup(state)
int heuristic_corner_db(const CubeState& state, const CornerPatternDB& corner_db);

// h = max(corner_db, edge_orient_db)
int heuristic_combined(const CubeState& state, const PatternDatabases& dbs);

// HeuristicFn closures for use with IDASolver
std::function<int(const CubeState&)> make_heuristic(const PatternDatabases& dbs);
std::function<int(const CubeState&)> make_heuristic_corner_only(const CornerPatternDB& db);
