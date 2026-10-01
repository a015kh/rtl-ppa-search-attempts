#!/bin/bash

# Base directories
EXP_DIR="exp"
SCRIPT="scripts/plot_problem_pareto.py"

# Create the base output directory
mkdir -p pareto

# Find all generation_log.jsonl files
find "$EXP_DIR" -type f -name "generation_log.jsonl" | while read -r GEN_LOG; do
    # Extract the directory containing generation_log.jsonl
    PROB_DIR=$(dirname "$GEN_LOG")
    
    # Extract problem name (basename of the directory)
    PROBLEM=$(basename "$PROB_DIR")
    
    # Extract benchmark name
    BENCHMARK_DIR=$(dirname "$PROB_DIR")
    BENCHMARK=$(basename "$BENCHMARK_DIR")
    
    # Extract model name
    MODEL_DIR=$(dirname "$BENCHMARK_DIR")
    MODEL=$(basename "$MODEL_DIR")
    
    # Define the summary JSON file
    SUMMARY_JSON="$PROB_DIR/${PROBLEM}_summary.json"
    
    # Check if summary JSON exists
    if [ ! -f "$SUMMARY_JSON" ]; then
        echo "Warning: Summary JSON $SUMMARY_JSON not found. Skipping $PROB_DIR"
        continue
    fi
    
    # Define output directory
    OUT_DIR="pareto/${MODEL}/${BENCHMARK}/${PROBLEM}"
    
    # Create problem specific output directory
    mkdir -p "$OUT_DIR"
    
    echo "Processing $MODEL / $BENCHMARK / $PROBLEM"
    
    # Run the plot script
    # We do not pass --max_area and --max_power so it uses the script's defaults
    python3 "$SCRIPT" \
        --generation_log "$GEN_LOG" \
        --summary_json "$SUMMARY_JSON" \
        --output_dir "$OUT_DIR"
done

echo "All plots generated in pareto/ directory."
