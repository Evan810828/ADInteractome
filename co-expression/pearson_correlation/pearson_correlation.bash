#!/bin/bash

# Define an array of cell types
cell_types=("cell_type_L5_ET" "cell_type_L6_CT" "cell_type_Pax6" "cell_type_L5_6_NP" "cell_type_L6b" "cell_type_Chandelier" "cell_type_L6_IT_Car3" "neuron_data")

# Loop over each cell type and run the task in parallel using background processes
for cell_type in "${cell_types[@]}"
do
    echo "Running task for cell type: ${cell_type}"
    python co-expression/pearson_correlation.py --data=data/split_train/"${cell_type}".h5ad &
done

# Wait for all background processes to finish
wait
