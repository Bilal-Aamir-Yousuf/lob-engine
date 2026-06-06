cmake_minimum_required(VERSION 3.16)
project(lob_engine LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options (later phases flip these on)
# ---------------------------------------------------------------------------
option(LOB_BUILD_TESTS      "Build GoogleTest unit tests"   ON)
option(LOB_BUILD_BENCHMARKS "Build Google Benchmark suite"  OFF)
option(LOB_BUILD_PYTHON     "Build the pybind11 module"     OFF)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

# ---------------------------------------------------------------------------
# Core engine library
# ---------------------------------------------------------------------------
add_library(lob_core
    src/order_book.cpp
    src/hawkes.cpp
    src/impact.cpp
    src/fix.cpp
)
target_include_directories(lob_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lob_core PUBLIC cxx_std_17)

# Required on Linux: this static lib is linked into the pybind11 shared module,
# and every object in a .so must be position-independent. No-op on MSVC.
set_target_properties(lob_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(lob_core PRIVATE /O2 /W4)
else()
    target_compile_options(lob_core PRIVATE -O3 -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Tests (GoogleTest via FetchContent)
# ---------------------------------------------------------------------------
if(LOB_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )
    FetchContent_MakeAvailable(benchmark)

    add_executable(bench_book benchmarks/bench_book.cpp)
    target_link_libraries(bench_book PRIVATE lob_core benchmark::benchmark)
endif()

if(LOB_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(lob_engine src/bindings.cpp)
    target_link_libraries(lob_engine PRIVATE lob_core)

    # Install next to the package root so scikit-build-core / pip can find it.
    install(TARGETS lob_engine DESTINATION .)
endif()

if(LOB_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    # Don't let gtest override our compiler/linker settings on Windows.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(test_book tests/test_book.cpp)
    target_link_libraries(test_book PRIVATE lob_core GTest::gtest_main)

    add_executable(test_hawkes tests/test_hawkes.cpp)
    target_link_libraries(test_hawkes PRIVATE lob_core GTest::gtest_main)

    add_executable(test_impact tests/test_impact.cpp)
    target_link_libraries(test_impact PRIVATE lob_core GTest::gtest_main)

    add_executable(test_fix tests/test_fix.cpp)
    target_link_libraries(test_fix PRIVATE lob_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(test_book)
    gtest_discover_tests(test_hawkes)
    gtest_discover_tests(test_impact)
    gtest_discover_tests(test_fix)
endif()
