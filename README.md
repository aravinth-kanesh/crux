# Crux

Optimal 3x3 Rubik's cube solver. Finds the shortest possible solution to any
scramble, guaranteed minimum move count (20 moves or fewer, per God's Number).

Implements IDA* (Iterative Deepening A\*) with pattern databases, based on
Richard Korf's 1997 paper *Finding Optimal Solutions to Rubik's Cube Using
Pattern Databases*.

## Algorithm

The solver uses `FastIDASolver`, a cache-friendly IDA* variant with O(1) index
updates via precomputed move tables. The heuristic is `max(h1, h2, h3, h4)`:

- **Corner pattern database** - minimum moves to solve all 8 corners for every
  corner configuration (8! x 3^7 = 88,179,840 states, ~42 MB on disk).
- **Edge partial database, group 0** - minimum moves to correctly place and
  orient edges UR, UF, UL, UB, DR, DF (P(12,6) x 2^6 = 42,577,920 states, ~21 MB).
- **Edge partial database, group 1** - same for edges DL, DB, FR, FL, BL, BR (~21 MB).
- **Edge partial database, group 2** - same for alternating edges UR, UL, DR, DL, FR, BL
  (~21 MB). Overlapping with both groups gives tighter bounds.

All four are admissible, so IDA* returns the minimum-move solution.

At search time, the 18 root moves are distributed across threads
(`ParallelFastIDASolver`): each threshold iteration runs 18 tasks in parallel,
and threads abort early once any thread finds a solution.

## Build

Requires CMake >= 3.16, a C++17 compiler, and POSIX threads.

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

## Usage

On first use, build the pattern databases (one-time, ~5-10 minutes):

```bash
mkdir -p data
./build-release/build_pattern_db --data-dir data
```

Then solve:

```bash
# Solve a specific scramble (prints coloured cube before and after)
./build-release/cube_solver --scramble "R U R' U' F2 D' L2 U"

# Solve a random scramble of given depth
./build-release/cube_solver --random-depth 15

# Skip pattern databases (much slower, useful for testing)
./build-release/cube_solver --scramble "R U F" --no-pattern-db

# Interactive mode: enter moves, type 'show' to display the cube, 'solve' to solve
./build-release/cube_solver --interactive
```

The solver prints an ANSI-coloured cross-layout of the cube before and after
each solve. Set `NO_COLOR=1` to fall back to letter codes (W/Y/G/B/O/R).

### Options

| Option | Description |
|--------|-------------|
| `--scramble "..."` | WCA-notation scramble to solve |
| `--random-depth N` | Generate and solve a random N-move scramble |
| `--seed N` | RNG seed for random scramble |
| `--data-dir DIR` | Pattern database directory (default: `./data`) |
| `--no-pattern-db` | Use simple heuristic instead of pattern databases |
| `--max-depth N` | Maximum solution depth to search (default: 20) |
| `--interactive` | Enter moves interactively, then call `solve` |

## Performance

Measured on Apple M-series (8-core, Release build, 4 pattern DBs, parallel solver):

| Scramble depth | Nodes explored | Solve time |
|----------------|----------------|------------|
| up to 10 moves | < 1,000 | < 2 ms |
| 12 moves | 4K - 46K | 3 - 25 ms (median 13 ms) |
| 15 moves | 9M - 88M | 5 - 49 s (median 33 s) |

Performance at depth 15 varies widely depending on how tight the heuristic
lower bound is for a given scramble. Worst-case scrambles (where all four
pattern DBs give a lower bound several moves below the true optimum) require
more IDA* iterations and dominate the runtime.

Without the pattern databases (`--no-pattern-db`), the misplaced-cubies heuristic
is far weaker. Expect 100-1000x slower for scrambles deeper than 10 moves.

## Tests

```bash
cmake -B build && cmake --build build
cd build && ctest --output-on-failure
```

32 tests covering move generation and encoding, IDA* correctness and optimality,
pattern database admissibility, and move-table correctness.

## Project Structure

```
src/
  cube.h / cube.cpp              - CubeState, composition, Lehmer encoding,
                                   encode_edge_partial
  moves.h / moves.cpp            - 18 moves, move tables, parsing, pruning
  solver.h / solver.cpp          - IDASolver (IDA* with pluggable heuristic)
  pattern_db.h / pattern_db.cpp  - CornerPatternDB, EdgePatternDB,
                                   PatternDatabases
  fast_solver.h / fast_solver.cpp - FastIDASolver, ParallelFastIDASolver,
                                    CP/CO/EO/INV_EP move tables
  heuristic.h / heuristic.cpp    - heuristic wrappers for IDASolver
  utils.h / utils.cpp            - validation, ANSI cube display, formatting
  main.cpp                       - CLI entry point
  build_db_main.cpp              - pattern database builder
tests/
  test_moves.cpp        - move generation and encoding
  test_solver.cpp       - IDA* correctness and optimality
  test_pattern_db.cpp   - pattern database admissibility
  test_fast_solver.cpp  - move-table correctness
  benchmark.cpp         - performance benchmark (not a CTest test)
data/                   - generated DB files (not committed)
  corner_pattern.db     - ~42 MB
  edge_orient.db        - ~2 KB
  edge1_pattern.db      - ~21 MB (edges UR UF UL UB DR DF)
  edge2_pattern.db      - ~21 MB (edges DL DB FR FL BL BR)
  edge3_pattern.db      - ~21 MB (alternating edges UR UL DR DL FR BL)
```

## 3D Visualiser

`visualise.py` calls the solver and animates the scramble and solution in a
3D OpenGL window. Each move is rendered with a smooth ease-in/ease-out
rotation, with a slowly orbiting camera and diffuse lighting.

Requires Python 3, numpy, pygame, and PyOpenGL:

```bash
pip install numpy pygame PyOpenGL PyOpenGL_accelerate
```

```bash
# Animate a specific scramble and its solution
python3 visualise.py --scramble "R U R' U' F2 D' L2 U" --data-dir data

# Animate a random 15-move scramble
python3 visualise.py --random-depth 15 --data-dir data

# Show scramble only, without solving
python3 visualise.py --scramble "R U R' U'" --no-solve
```

Press `Esc` or `Q` to quit.

## References

- Korf, R. E. (1997). *Finding Optimal Solutions to Rubik's Cube Using Pattern
  Databases*. AAAI/IAAI.
- Rokicki, T. et al. (2010). *God's Number is 20*.
  [cube20.org](https://cube20.org)
