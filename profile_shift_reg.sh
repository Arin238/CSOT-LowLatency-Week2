#!/bin/bash
# Exit on error
set -e

# Build directory for profiling
BUILD_DIR="build-shift-reg"
TRACE_FILE="data/large.trace"

# Default action is running full stats
ACTION=${1:-stats}

# Ensure trace file exists or warn the user
if [ ! -f "$TRACE_FILE" ]; then
    echo "Warning: $TRACE_FILE not found."
    echo "To generate a real large trace run:"
    echo "  python3 data/gen_trace.py --accesses 5000000 --seed 42 --out data/large.trace"
    echo "Falling back to generating a small 10,000-access trace for debugging..."
    mkdir -p data
    python3 data/gen_trace.py --accesses 10000 --seed 42 --out "$TRACE_FILE"
fi

# Helper function to build the binary
# $1 = build type (Debug or Release)
# $2 = CSOT_CHECK_ALLOCS (ON or OFF)
build_binary() {
    local build_type=$1
    local check_allocs=$2
    echo "=== Building Cache Sim Runner (Type=$build_type, CheckAllocs=$check_allocs) ==="
    # Wipe cmake cache to avoid stale variable issues
    rm -f "$BUILD_DIR/CMakeCache.txt"
    cmake -B "$BUILD_DIR" \
          -DCMAKE_BUILD_TYPE="$build_type" \
          -DCSOT_CACHE_SIM_SRC=cache_shift_reg.cpp \
          -DCSOT_CHECK_ALLOCS="$check_allocs" \
          -DENABLE_LTO=OFF
    cmake --build "$BUILD_DIR" -j
}

case "$ACTION" in
    "check")
        # Build in Debug (-O0) so the compiler cannot elide allocations
        build_binary "Debug" "ON"
        echo "=== Running Hot Path Allocation Verification ==="
        echo "If any heap allocation (new/malloc) occurs on the hot path, the program will crash immediately."
        ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"
        echo "Success! Zero heap allocations detected on the hot path."
        ;;

    "stats")
        build_binary "Release" "OFF"
        echo "=== Collecting Hardware PMU Performance Counters (perf stat) ==="
        # Record key CPU and Cache counters for low-latency tuning
        perf stat -d -- \

            taskset -c 0 ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"
        ;;

    "record")
        build_binary "Release" "OFF"
        echo "=== Recording CPU Cycles (perf record) ==="
        echo "Recording call graphs at 999Hz sample rate. Use 'perf report' afterward to inspect."
        perf record -F 999 -g --call-graph dwarf -- \
            taskset -c 0 ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"
        
        echo "=== Opening Report ==="
        perf report -g
        ;;

    "malloc")
        build_binary "Debug" "ON"
        # Find libc path dynamically
        LIBC_PATH=$(ldd ./"$BUILD_DIR"/cache_sim_runner | grep libc.so | awk '{print $3}')
        if [ -z "$LIBC_PATH" ]; then
            LIBC_PATH="/lib/x86_64-linux-gnu/libc.so.6"
        fi

        echo "=== Registering libc:malloc probe ==="
        sudo perf probe -d probe_libc:malloc 2>/dev/null || true
        sudo perf probe -x "$LIBC_PATH" malloc

        echo "=== Recording malloc activations on the Hot Path ==="
        # --delay=-1 relies on prctl() controls inside cache_sim.cpp to only trace the hot path
        sudo perf record -e probe_libc:malloc -g --delay=-1 -- \
            ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"

        echo "=== Displaying Allocation Report ==="
        sudo perf report -g
        ;;

    "annotate")
        build_binary "Release" "OFF"
        echo "=== Recording CPU Cycles for Annotation ==="
        # Record at maximum frequency without callgraphs for precise instruction profiling
        perf record -F max -- \
            taskset -c 0 ./"$BUILD_DIR"/cache_sim_runner "$TRACE_FILE"
        
        echo "=== Opening Perf Annotate ==="
        # Use intel syntax and show source interleaving
        perf annotate -M intel --source
        ;;

    *)
        echo "Usage: $0 [check | stats | record | malloc | annotate]"
        echo "  check    : Assert zero heap allocations on the hot path"
        echo "  stats    : Print hardware PMU counters"
        echo "  record   : Sample CPU cycles and open a call-graph report"
        echo "  malloc   : Hook into malloc calls to track active allocation call stacks"
        echo "  annotate : Record and open instruction-level assembly annotation"
        exit 1
        ;;
esac
