#!/usr/bin/env bash

# Parallel Benchmark Script for Rust Examples
# Requires GNU Parallel

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
TIMEOUT=15  # seconds time limit
SUPPORT=1
DEPTHS=(3 4 5 6)  # Depths from 4 to 9
NBTHREAD=20  # Default number of parallel jobs
TEST_DATA_DIR="test_data"
BASE_RESULT_DIR="results_new_imp"

# Algorithm-specific parameters
PURITY_EPSILON=0.002
LDS_DISCREPANCY_TYPES=("monotonic" "exponential" "luby")
CACHE_TYPES=("trie" "hashmap")
LAMBDA_VALUES=(0 1 3 5 10)
FAST_D2="enabled"

# Argument parsing with validation
parse_arguments() {
    local scripts=()
    local custom_input_dir=""
    local custom_output_dir=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --purity) scripts+=("purity") ;;
            --gain) scripts+=("gain") ;;
            --lds) scripts+=("lds") ;;
            --topk) scripts+=("topk") ;;
            --restart) scripts+=("restart") ;;
            --gainlds) scripts+=("gainlds") ;;
            --gaintopk) scripts+=("gaintopk") ;;
            --lookahead) scripts+=("lookahead") ;;
            --lookahead-lds) scripts+=("lookahead_lds") ;;
            --lookahead-topk) scripts+=("lookahead_topk") ;;
            --all) scripts=("purity" "gain" "lds" "topk" "restart" "gainlds" "gaintopk" "lookahead" "lookahead_lds" "lookahead_topk") ;;
            --cache-types)
                IFS=',' read -ra CACHE_TYPES <<< "$2"
                shift ;;
            --lambda-values)
                IFS=',' read -ra LAMBDA_VALUES <<< "$2"
                shift ;;
            --fast-d2)
                FAST_D2="$2"
                shift ;;
            --input-dir)
                custom_input_dir="$2"
                shift ;;
            --output-dir)
                custom_output_dir="$2"
                shift ;;
            --threads)
                NBTHREAD="$2"
                shift ;;
            --timeout)
                TIMEOUT="$2"
                shift ;;
            --dry-run) dry_run=true ;;
            *) error_exit "Unknown parameter: $1" ;;
        esac
        shift
    done

    # Use custom directories if provided
    [[ -n "$custom_input_dir" ]] && TEST_DATA_DIR="$custom_input_dir"
    [[ -n "$custom_output_dir" ]] && BASE_RESULT_DIR="$custom_output_dir"

    # Validate input directory
    [[ ! -d "$TEST_DATA_DIR" ]] && error_exit "Input directory not found: $TEST_DATA_DIR"

    # Generate input files list
    INPUT_FILES=($(find "$TEST_DATA_DIR" -type f -name "*.txt"))

    # Validate script selection
    [[ ${#scripts[@]} -eq 0 ]] && scripts=("purity" "gain" "lds" "topk" "restart" "gainlds" "gaintopk" "lookahead" "lookahead_lds" "lookahead_topk")

    # Return global variables
    SELECTED_SCRIPTS=("${scripts[@]}")
}

# Function to run an individual benchmark
run_benchmark() {
    local algo="$1"
    local dataset="$2"
    local depth="$3"
    local output_dir="$4"
    local extra_param="$5"
    local cache_type="$6"
    local lambda="$7"

    local dataset_name=$(basename "$dataset")

    # Build command with algorithm-specific parameters
    local cmd="cargo run --release --example $algo -- --input $dataset --depth $depth --support $SUPPORT --timeout $TIMEOUT --heuristic information-gain --result $output_dir --fast-d2 $FAST_D2 --cache-type $cache_type --lambda $lambda --overwrite"

    # Add algorithm-specific parameters
    case "$algo" in
        "purity")
            cmd="$cmd --epsilon $PURITY_EPSILON"
            log "Running $algo on $dataset_name with depth $depth, cache=$cache_type, lambda=$lambda"
            ;;
        "gain")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "lds")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "topk")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "gaintopk")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "gainlds")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "lookahead")
            log "Running $algo on $dataset_name with depth $depth, cache=$cache_type, lambda=$lambda"
            ;;
        "lookahead_lds")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "lookahead_topk")
            cmd="$cmd --step $extra_param"
            log "Running $algo on $dataset_name with depth $depth, discrepancy=$extra_param, cache=$cache_type, lambda=$lambda"
            ;;
        "restart")
            log "Running $algo on $dataset_name with depth $depth, cache=$cache_type, lambda=$lambda"
            ;;
    esac

    if [[ "$dry_run" == true ]]; then
        echo "DRY RUN: $cmd"
        return 0
    fi

    # Run the actual benchmark
    eval "$cmd"
}

# Export the function for parallel
export -f run_benchmark
export -f log
export -f error_exit
export TIMEOUT SUPPORT dry_run PURITY_EPSILON FAST_D2

# Main execution function
run_benchmarks() {
    # Create main timestamp-based output directory
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="${BASE_RESULT_DIR}/results_${TIMEOUT}"
    mkdir -p "$output_dir"

    log "Starting benchmarks with timeout=$TIMEOUT, depths=${DEPTHS[*]}, threads=$NBTHREAD"
    log "Purity epsilon: $PURITY_EPSILON"
    log "Discrepancy types: ${LDS_DISCREPANCY_TYPES[*]}"
    log "Cache types: ${CACHE_TYPES[*]}"
    log "Lambda values: ${LAMBDA_VALUES[*]}"
    log "Fast-D2: $FAST_D2"
    log "Input directory: $TEST_DATA_DIR"
    log "Output directory: $output_dir"

    # Check if GNU Parallel is installed
    if ! command -v parallel &> /dev/null; then
        error_exit "GNU Parallel is required but not installed"
    fi

    # Create command file for parallel
    CMDFILE=$(mktemp)

    # Build all combinations
    for script in "${SELECTED_SCRIPTS[@]}"; do
        for input_file in "${INPUT_FILES[@]}"; do
            for depth in "${DEPTHS[@]}"; do
                for cache_type in "${CACHE_TYPES[@]}"; do
                    for lambda in "${LAMBDA_VALUES[@]}"; do
                        case "$script" in
                            "lds"|"gain"|"topk"|"gainlds"|"gaintopk"|"lookahead_lds"|"lookahead_topk")
                                for discrepancy in "${LDS_DISCREPANCY_TYPES[@]}"; do
                                    echo "run_benchmark $script $input_file $depth $output_dir $discrepancy $cache_type $lambda" >> "$CMDFILE"
                                done
                                ;;
                            *)
                                # lookahead, purity, restart: no discrepancy parameter
                                echo "run_benchmark $script $input_file $depth $output_dir '' $cache_type $lambda" >> "$CMDFILE"
                                ;;
                        esac
                    done
                done
            done
        done
    done

    # Run with parallel
    log "Running $(wc -l < "$CMDFILE") benchmark combinations with $NBTHREAD parallel jobs"
    parallel --bar --progress --joblog "${output_dir}/parallel.log" -j "$NBTHREAD" < "$CMDFILE"

    # Clean up
    rm "$CMDFILE"

    log "Benchmarks completed. Results stored in: $output_dir"
}

# Main script execution
main() {
    parse_arguments "$@"
    run_benchmarks
}

# Execute main with all arguments
main "$@"
