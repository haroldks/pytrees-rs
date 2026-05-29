#!/usr/bin/env bash

# Multi-Timeout Benchmark Wrapper Script
# Runs the LDS benchmark script with multiple timeout values

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Error handling function
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Default configuration
BENCHMARK_SCRIPT="./benchmarks.sh"
TIMEOUT_VALUES=(5 15 30 60 120 150)  # Default timeout values in seconds
BASE_ARGS=""``

# Argument parsing
parse_arguments() {
    local custom_timeouts=()
    local passthrough_args=()

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --timeouts)
                IFS=',' read -ra custom_timeouts <<< "$2"
                shift ;;
            --benchmark-script)
                BENCHMARK_SCRIPT="$2"
                shift ;;
            --help)
                show_help
                exit 0 ;;
            *)
                # Collect all other arguments to pass through
                # Skip --timeout if present (we'll override it)
                if [[ "$1" == "--timeout" ]]; then
                    shift  # Skip the --timeout flag
                    shift  # Skip the timeout value
                    continue
                fi
                passthrough_args+=("$1")
                ;;
        esac
        shift
    done

    # Use custom timeouts if provided
    [[ ${#custom_timeouts[@]} -gt 0 ]] && TIMEOUT_VALUES=("${custom_timeouts[@]}")

    # Build base arguments string
    BASE_ARGS="${passthrough_args[*]}"
}

# Show help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] [BENCHMARK_OPTIONS]

Wrapper script to run LDS benchmarks with multiple timeout values.

WRAPPER OPTIONS:
    --timeouts T1,T2,...     Comma-separated list of timeout values in seconds
                             (default: 5,10,30,60,120,300,600)
    --benchmark-script PATH  Path to the benchmark script (default: ./benchmark_lds.sh)
    --help                   Show this help message

BENCHMARK_OPTIONS:
    All other options are passed directly to the benchmark script.
    Note: --timeout will be ignored as it's controlled by --timeouts.

EXAMPLES:
    # Run with default timeouts and custom benchmark options
    $0 --input-dir test_data --threads 3 --heuristic-only

    # Run with custom timeout values
    $0 --timeouts 10,60,300 --input-dir test_data --depths 3,4,5

    # Run with specific configuration
    $0 --timeouts 5,15,30 --input-dir test_data --output-dir results \\
       --heuristic-only --strategy first --depths 3,4

    # Use custom benchmark script location
    $0 --benchmark-script ./scripts/benchmark_lds.sh --timeouts 10,30

TIMEOUT VALUES:
    Default: 5, 10, 30, 60, 120, 300, 600 seconds
    Custom: Use --timeouts to specify your own comma-separated list
EOF
}

# Main execution function
run_multi_timeout_benchmarks() {
    log "=========================================="
    log "Multi-Timeout Benchmark Configuration"
    log "=========================================="
    log "Benchmark script: $BENCHMARK_SCRIPT"
    log "Timeout values: ${TIMEOUT_VALUES[*]}"
    log "Base arguments: $BASE_ARGS"
    log "Total runs: ${#TIMEOUT_VALUES[@]}"
    log "=========================================="

    # Check if benchmark script exists
    [[ ! -f "$BENCHMARK_SCRIPT" ]] && error_exit "Benchmark script not found: $BENCHMARK_SCRIPT"
    [[ ! -x "$BENCHMARK_SCRIPT" ]] && error_exit "Benchmark script is not executable: $BENCHMARK_SCRIPT"

    local run_number=1
    local total_runs=${#TIMEOUT_VALUES[@]}
    local failed_runs=()

    for timeout in "${TIMEOUT_VALUES[@]}"; do
        log ""
        log "=========================================="
        log "Run $run_number/$total_runs: Timeout = ${timeout}s"
        log "=========================================="

        # Build the full command
        local cmd="$BENCHMARK_SCRIPT --timeout $timeout $BASE_ARGS"

        log "Executing: $cmd"

        # Run the benchmark
        if eval "$cmd"; then
            log "Run $run_number/$total_runs completed successfully (timeout=${timeout}s)"
        else
            local exit_code=$?
            log "WARNING: Run $run_number/$total_runs failed (timeout=${timeout}s, exit code: $exit_code)"
            failed_runs+=("${timeout}s")
        fi

        ((run_number++))

        # Add a small delay between runs
        if [[ $run_number -le $total_runs ]]; then
            log "Waiting 2 seconds before next run..."
            sleep 2
        fi
    done

    log ""
    log "=========================================="
    log "Multi-Timeout Benchmark Summary"
    log "=========================================="
    log "Total runs: $total_runs"
    log "Successful: $((total_runs - ${#failed_runs[@]}))"
    log "Failed: ${#failed_runs[@]}"

    if [[ ${#failed_runs[@]} -gt 0 ]]; then
        log "Failed timeout values: ${failed_runs[*]}"
    fi

    log "=========================================="

    # Return non-zero if any runs failed
    [[ ${#failed_runs[@]} -gt 0 ]] && return 1
    return 0
}

# Main script execution
main() {
    parse_arguments "$@"
    run_multi_timeout_benchmarks
}

# Execute main with all arguments
main "$@"
