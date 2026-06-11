#!/bin/bash
set -e

BUILD_DIR="build-gen-report-new"
TRACE_FILE="data/large.trace"

# Ensure trace file exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Warning: $TRACE_FILE not found. Falling back to a small trace..."
    mkdir -p data
    python3 data/gen_trace.py --accesses 10000 --seed 42 --out "$TRACE_FILE"
fi

echo "=== Building cache_sim_new.cpp ==="
rm -f "$BUILD_DIR/CMakeCache.txt"
cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCSOT_CACHE_SIM_SRC=cache_sim_new.cpp \
      -DCSOT_CHECK_ALLOCS=OFF \
      -DENABLE_LTO=OFF
cmake --build "$BUILD_DIR" -j

echo "=== Recording perf data ==="
perf record -F max -g --call-graph dwarf -- \
    taskset -c 0 ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"

echo "=== Generating perf_report_new.txt ==="
perf report --stdio > perf_report_new.txt

echo "=== Opening Perf Annotate ==="
# Notice we dropped the exact mangled name because it's different for the new class.
# This will automatically open the hottest function.
perf annotate -M intel --source
