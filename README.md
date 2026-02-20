# Optimal Rubik's Cube Solver

Finds provably optimal solutions to any 3×3 Rubik's cube scramble - guaranteed
minimum move count (≤20 moves, by God's Number).

Implements IDA* (Iterative Deepening A\*) with pattern databases, based on
Richard Korf's 1997 paper *Finding Optimal Solutions to Rubik's Cube Using
Pattern Databases*.

## Algorithm

The solver uses IDA* with two heuristics combined via `max()`:

- **Corner pattern database** - stores the minimum moves to solve all 8 corners
  for every possible corner configuration (8! × 3⁷ = 88,179,840 states, ~42 MB).
  Built once via BFS from the solved state and saved to disk.
- **Edge orientation database** - stores the minimum moves to correctly orient
  all 12 edges (2¹¹ = 2048 states, ~2 KB).

Both are admissible heuristics (never overestimate), so IDA* finds optimal solutions.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler.

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
# Solve a specific scramble
./build-release/cube_solver --scramble "R U R' U' F2 D' L2 U"

# Solve a random scramble of given depth
./build-release/cube_solver --random-depth 15

# Skip pattern databases (much slower, useful for testing)
./build-release/cube_solver --scramble "R U F" --no-pattern-db

# Interactive mode
./build-release/cube_solver --interactive
```

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

With the corner pattern database loaded:

| Scramble depth | Typical solve time |
|----------------|--------------------|
| 10 moves | < 0.01s |
| 15 moves | < 1s |
| 18 moves | 10–60s |
| 20 moves | minutes |

Without the pattern database (`--no-pattern-db`), the misplaced-cubies heuristic
is much weaker - expect 10–100× slower for deep scrambles.

## Tests

```bash
ctest --output-on-failure
```

28 tests covering move correctness, IDA* optimality, and pattern database
admissibility.

## Project Structure

```
src/
  cube.h / cube.cpp         - cube state, composition, Lehmer encoding
  moves.h / moves.cpp       - 18 moves, move tables, parsing, pruning
  solver.h / solver.cpp     - IDA* implementation
  pattern_db.h / pattern_db.cpp  - corner and edge-orient pattern databases
  heuristic.h / heuristic.cpp    - heuristic wrappers
  utils.h / utils.cpp       - validation, formatting
  main.cpp                  - CLI entry point
  build_db_main.cpp         - pattern database builder
tests/
  test_moves.cpp            - move generation and encoding tests
  test_solver.cpp           - solver correctness and optimality tests
  test_pattern_db.cpp       - pattern database admissibility tests
```

## References

- Korf, R. E. (1997). *Finding Optimal Solutions to Rubik's Cube Using Pattern
  Databases*. AAAI/IAAI.
- Rokicki, T. et al. (2010). *God's Number is 20*.
  [cube20.org](https://cube20.org)
