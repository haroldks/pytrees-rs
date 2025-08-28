#!/bin/bash

# Script to run the purity example with multiple depth values
# Usage: ./depth_runner.sh [start_depth] [end_depth] [step] [additional_args...]

# Check if enough arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 [start_depth] [end_depth] [step] [additional_args...]"
    echo "Example: $0 1 10 2 --support 1 --timeout 300"
    exit 1
fi

COMMAND="target/release/examples/gain"
INPUT_FILE="test_data/anneal.txt"
START_DEPTH="$1"
END_DEPTH="$2"
STEP="$3"
shift 3  # Remove the first 3 arguments

# Check if command exists
if [ ! -f "$COMMAND" ]; then
    echo "Error: Command '$COMMAND' not found"
    exit 1
fi

# Run the command for each depth
echo "Running $COMMAND for depths from $START_DEPTH to $END_DEPTH with step $STEP"
for ((depth = START_DEPTH; depth <= END_DEPTH; depth += STEP)); do
    echo "-------------------------------------------"
    echo "Executing: $COMMAND --input $INPUT_FILE --depth $depth --support 1 --timeout 300 --metric 1.0 --epsilon 0.002 --result . $@"
    echo "-------------------------------------------"

    $COMMAND --input "$INPUT_FILE" --depth "$depth" --support 1 --timeout 300 --metric 1.0 --epsilon 0.002 --result . "$@"

    # Check if command executed successfully
    if [ $? -ne 0 ]; then
        echo "Command failed with depth=$depth"
    fi

    echo "" # Add a blank line for readability
done

echo "All runs completed!"
