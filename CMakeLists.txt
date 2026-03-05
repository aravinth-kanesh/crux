cmake_minimum_required(VERSION 3.16)
project(RubiksSolver VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimisation flags for release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Core solver library
add_library(cube_lib STATIC
    src/cube.cpp
    src/moves.cpp
    src/solver.cpp
    src/pattern_db.cpp
    src/heuristic.cpp
    src/utils.cpp
    src/fast_solver.cpp
)
target_include_directories(cube_lib PUBLIC src)

# Main executable
add_executable(cube_solver src/main.cpp)
target_link_libraries(cube_solver cube_lib)

# Pattern DB builder tool
add_executable(build_pattern_db src/build_db_main.cpp)
target_link_libraries(build_pattern_db cube_lib)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_moves tests/test_moves.cpp)
    target_link_libraries(test_moves cube_lib)
    add_test(NAME test_moves COMMAND test_moves)

    add_executable(test_solver tests/test_solver.cpp)
    target_link_libraries(test_solver cube_lib)
    add_test(NAME test_solver COMMAND test_solver)

    add_executable(test_pattern_db tests/test_pattern_db.cpp)
    target_link_libraries(test_pattern_db cube_lib)
    add_test(NAME test_pattern_db COMMAND test_pattern_db)

    add_executable(test_fast_solver tests/test_fast_solver.cpp)
    target_link_libraries(test_fast_solver cube_lib)
    add_test(NAME test_fast_solver COMMAND test_fast_solver)

    add_executable(benchmark tests/benchmark.cpp)
    target_link_libraries(benchmark cube_lib)
endif()
