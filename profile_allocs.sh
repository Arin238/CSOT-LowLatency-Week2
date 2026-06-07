#!/bin/bash
# Exit on error
set -e

# Build directory for allocation checks
BUILD_DIR="build-allocs"
TRACE_FILE="data/tiny.trace"

# Default action is checking allocations
ACTION=${1:-check}

# Ensure trace file exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Warning: $TRACE_FILE not found, generating tiny trace..."
    python3 data/gen_trace.py --accesses 1000 --seed 42 --out "$TRACE_FILE"
fi

if [ "$ACTION" = "check" ]; then
    echo "=== Building with Hot Path Allocation Assertions ==="
    cmake -B "$BUILD_DIR" -DCSOT_CACHE_SIM_SRC=cache_sim.cpp -DCSOT_CHECK_ALLOCS=ON
    cmake --build "$BUILD_DIR" -j

    echo "=== Running Cache Sim Runner ==="
    echo "If any heap allocation is triggered on the hot path, the program will crash/abort immediately."
    ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"
    echo "Success! No heap allocations detected on the hot path."

elif [ "$ACTION" = "perf-malloc" ]; then
    echo "=== Building executable for profiling ==="
    cmake -B "$BUILD_DIR" -DCSOT_CACHE_SIM_SRC=cache_sim.cpp -DCSOT_CHECK_ALLOCS=ON
    cmake --build "$BUILD_DIR" -j

    # Find libc library path dynamically
    LIBC_PATH=$(ldd ./"$BUILD_DIR"/cache_sim_runner | grep libc.so | awk '{print $3}')
    if [ -z "$LIBC_PATH" ]; then
        # fallback to standard location
        LIBC_PATH="/lib/x86_64-linux-gnu/libc.so.6"
    fi

    echo "Using libc at: $LIBC_PATH"

    echo "=== Adding perf probe for malloc ==="
    # Remove existing probe if it exists
    sudo perf probe -d probe_libc:malloc 2>/dev/null || true
    sudo perf probe -x "$LIBC_PATH" malloc

    echo "=== Recording malloc allocations ==="
    # We use delay=-1 because prctl(PR_TASK_PERF_EVENTS_ENABLE/DISABLE) will control it programmatically on the hot path
    sudo perf record -e probe_libc:malloc -g --delay=-1 -- ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"

    echo "=== Displaying report ==="
    sudo perf report -g

elif [ "$ACTION" = "perf-cpu" ]; then
    echo "=== Building executable for CPU profiling ==="
    cmake -B "$BUILD_DIR" -DCSOT_CACHE_SIM_SRC=cache_sim.cpp -DCSOT_CHECK_ALLOCS=OFF
    cmake --build "$BUILD_DIR" -j

    echo "=== Recording CPU cycles with DWARF stacks ==="
    perf record -F 999 -g --call-graph dwarf -- ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"

    echo "=== Displaying CPU report ==="
    perf report -g
else
    echo "Usage: $0 [check | perf-malloc | perf-cpu]"
    exit 1
fi
