#!/bin/bash
set -e

BUILD_DIR="build-gen-report"
TRACE_FILE="data/large.trace"

# Ensure trace file exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Warning: $TRACE_FILE not found. Falling back to a small trace..."
    mkdir -p data
    python3 data/gen_trace.py --accesses 10000 --seed 42 --out "$TRACE_FILE"
fi

echo "=== Building cache_sim.cpp ==="
rm -f "$BUILD_DIR/CMakeCache.txt"
cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCSOT_CACHE_SIM_SRC=cache_sim.cpp \
      -DCSOT_CHECK_ALLOCS=OFF \
      -DENABLE_LTO=OFF
cmake --build "$BUILD_DIR" -j

echo "=== Recording perf data ==="
perf record -F 999 -g --call-graph dwarf -- \
    taskset -c 0 ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"

echo "=== Generating perf_report.txt ==="
perf report --stdio > perf_report.txt

echo "=== Opening in nano ==="
nano perf_report.txt
