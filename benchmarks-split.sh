#!/usr/bin/env bash

# Parallel Benchmark Script for Split Examples

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
CACHE_TYPES=("trie" "hashmap")
FAST_D2="enabled"
NBTHREAD=70
TEST_DATA_DIR="test_data"
BASE_RESULT_DIR="results"
dry_run=false

parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --input-dir)
                TEST_DATA_DIR="$2"
                shift ;;
            --output-dir)
                BASE_RESULT_DIR="$2"
                shift ;;
            --threads)
                NBTHREAD="$2"
                shift ;;
            --timeout)
                TIMEOUT="$2"
                shift ;;
            --cache-types)
                IFS=',' read -ra CACHE_TYPES <<< "$2"
                shift ;;
            --lambda-values)
                IFS=',' read -ra LAMBDA_VALUES <<< "$2"
                shift ;;
            --fast-d2)
                FAST_D2="$2"
                shift ;;
            --dry-run)
                dry_run=true ;;
            *)
                error_exit "Unknown parameter: $1" ;;
        esac
        shift
    done

    [[ ! -d "$TEST_DATA_DIR" ]] && error_exit "Input directory not found: $TEST_DATA_DIR"

    mapfile -t INPUT_FILES < <(find "$TEST_DATA_DIR" -type f -name "*.txt")

    [[ ${#INPUT_FILES[@]} -eq 0 ]] && error_exit "No .txt files found in: $TEST_DATA_DIR"
}

run_benchmark() {
    local dataset="$1"
    local depth="$2"
    local lookahead="$3"
    local lambda="$4"
    local cache_type="$5"
    local output_dir="$6"

    local dataset_name
    dataset_name=$(basename "$dataset")

    log "Running split on $dataset_name | depth=$depth lookahead=$lookahead lambda=$lambda cache=$cache_type fast-d2=$FAST_D2"

    if [[ "$dry_run" == true ]]; then
        echo "DRY RUN: cargo run --release --example split -- --input $dataset --depth $depth --lookahead-depth $lookahead --lambda $lambda --support $SUPPORT --timeout $TIMEOUT --heuristic information-gain --cache-type $cache_type --fast-d2 $FAST_D2 --result $output_dir"
        return 0
    fi

    cargo run --release --example split -- \
        --input "$dataset" \
        --depth "$depth" \
        --lookahead-depth "$lookahead" \
        --lambda "$lambda" \
        --support "$SUPPORT" \
        --timeout "$TIMEOUT" \
        --heuristic information-gain \
        --cache-type "$cache_type" \
        --fast-d2 "$FAST_D2" \
        --result "$output_dir"
}

export -f run_benchmark
export -f log
export -f error_exit
export TIMEOUT SUPPORT dry_run FAST_D2

run_benchmarks() {
    local output_dir="${BASE_RESULT_DIR}/results_${TIMEOUT}"
    mkdir -p "$output_dir"

    log "Starting benchmarks"
    log "Depths: ${DEPTHS[*]}, Lookahead: 1 to depth-1"
    log "Lambda values: ${LAMBDA_VALUES[*]}"
    log "Cache types: ${CACHE_TYPES[*]}"
    log "Fast-D2: $FAST_D2"
    log "Threads: $NBTHREAD, Timeout: $TIMEOUT"
    log "Input: $TEST_DATA_DIR → Output: $output_dir"

    command -v parallel &> /dev/null || error_exit "GNU Parallel is required but not installed"

    local CMDFILE
    CMDFILE=$(mktemp)

    for input_file in "${INPUT_FILES[@]}"; do
        for depth in "${DEPTHS[@]}"; do
            for (( lookahead=1; lookahead<=depth-1; lookahead++ )); do
                for lambda in "${LAMBDA_VALUES[@]}"; do
                    for cache_type in "${CACHE_TYPES[@]}"; do
                        printf 'run_benchmark %q %q %q %q %q %q\n' \
                            "$input_file" \
                            "$depth" \
                            "$lookahead" \
                            "$lambda" \
                            "$cache_type" \
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

main "$@">?
