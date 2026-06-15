#!/usr/bin/env bash

# Parallel Benchmark Script for Split and HashSplit Examples

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Default configuration
TIMEOUT=180
SUPPORT=1
DEPTHS=(2 3 4 5 6)
LAMBDA_VALUES=(0 1 3 5 7 10)
EXAMPLES=("split" "hashsplit")
NBTHREAD=70
TEST_DATA_DIR="test_data"
BASE_RESULT_DIR="results"
dry_run=false

parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --input-dir)
                TEST_DATA_DIR="$2"
                shift
                ;;
            --output-dir)
                BASE_RESULT_DIR="$2"
                shift
                ;;
            --threads)
                NBTHREAD="$2"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift
                ;;
            --dry-run)
                dry_run=true
                ;;
            *)
                error_exit "Unknown parameter: $1"
                ;;
        esac
        shift
    done

    [[ ! -d "$TEST_DATA_DIR" ]] && error_exit "Input directory not found: $TEST_DATA_DIR"

    mapfile -t INPUT_FILES < <(find "$TEST_DATA_DIR" -type f -name "*.txt")

    [[ ${#INPUT_FILES[@]} -eq 0 ]] && error_exit "No .txt files found in: $TEST_DATA_DIR"
}

run_benchmark() {
    local example="$1"
    local dataset="$2"
    local depth="$3"
    local lookahead="$4"
    local lambda="$5"
    local output_dir="$6"

    local dataset_name
    dataset_name=$(basename "$dataset")

    local example_output_dir="${output_dir}/${example}"
    mkdir -p "$example_output_dir"

    log "Running $example on $dataset_name | depth=$depth lookahead=$lookahead lambda=$lambda"

    if [[ "$dry_run" == true ]]; then
        echo "DRY RUN: cargo run --release --example $example -- --input $dataset --depth $depth --lookahead-depth $lookahead --lambda $lambda --support $SUPPORT --timeout $TIMEOUT --heuristic information-gain --result $example_output_dir"
        return 0
    fi

    cargo run --release --example "$example" -- \
        --input "$dataset" \
        --depth "$depth" \
        --lookahead-depth "$lookahead" \
        --lambda "$lambda" \
        --support "$SUPPORT" \
        --timeout "$TIMEOUT" \
        --heuristic information-gain \
        --result "$example_output_dir"
}

export -f run_benchmark
export -f log
export -f error_exit
export TIMEOUT SUPPORT dry_run

run_benchmarks() {
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local output_dir="${BASE_RESULT_DIR}/results_${timestamp}"
    mkdir -p "$output_dir"

    log "Starting benchmarks"
    log "Examples: ${EXAMPLES[*]}"
    log "Depths: ${DEPTHS[*]}, Lookahead: 1 to depth-1, Lambda values: ${LAMBDA_VALUES[*]}"
    log "Threads: $NBTHREAD, Timeout: $TIMEOUT"
    log "Input: $TEST_DATA_DIR → Output: $output_dir"

    command -v parallel &> /dev/null || error_exit "GNU Parallel is required but not installed"

    local CMDFILE
    CMDFILE=$(mktemp)

    for example in "${EXAMPLES[@]}"; do
        for input_file in "${INPUT_FILES[@]}"; do
            for depth in "${DEPTHS[@]}"; do
                for (( lookahead=1; lookahead<=depth-1; lookahead++ )); do
                    for lambda in "${LAMBDA_VALUES[@]}"; do
                        printf 'run_benchmark %q %q %q %q %q %q\n' \
                            "$example" \
                            "$input_file" \
                            "$depth" \
                            "$lookahead" \
                            "$lambda" \
                            "$output_dir" >> "$CMDFILE"
                    done
                done
            done
        done
    done

    local total
    total=$(wc -l < "$CMDFILE")

    log "Total combinations: $total"

    parallel --bar --progress --joblog "${output_dir}/parallel.log" -j "$NBTHREAD" < "$CMDFILE"

    rm "$CMDFILE"

    log "Benchmarks completed. Results in: $output_dir"
}

main() {
    parse_arguments "$@"
    run_benchmarks
}

main "$@"
