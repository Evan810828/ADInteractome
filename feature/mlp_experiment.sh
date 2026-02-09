#!/bin/bash

# Define an array of cell types
cell_types=('Chandelier'
        'L2_3_IT'
        'L4_IT'
        'L5_ET'
        'L5_IT'
        'L5_6_NP'
        'L6_IT'
        'L6_CT'
        'L6b'
        'L6_IT_Car3'
        'Lamp5'
        'Pax6'
        'Vip'
        'Sst_Chodl'
        'Sst'
        'Pvalb'
        'Lamp5_Lhx6'
        'Sncg')

# Limit the number of parallel jobs
MAX_JOBS=3

# Loop over each cell type
for cell_type in "${cell_types[@]}"; do
    python feature/mlp_ranking.py --cell_type="${cell_type}" &

    # Check how many jobs are running, and wait if max reached
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

# Wait for all remaining background jobs to finish
wait
