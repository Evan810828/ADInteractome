#!/bin/bash

# Define an array of cell types
cell_types=('cell_type_Chandelier.h5ad',
        'cell_type_L2_3_IT.h5ad',
        'cell_type_L4_IT.h5ad',
        'cell_type_L5_ET.h5ad',
        'cell_type_L5_IT.h5ad',
        'cell_type_L5_6_NP.h5ad',
        'cell_type_L6_IT.h5ad',
        'cell_type_L6_CT.h5ad',
        'cell_type_L6b.h5ad',
        'cell_type_L6_IT_Car3.h5ad',
        'cell_type_Lamp5.h5ad',
        'cell_type_Pax6.h5ad',
        'cell_type_Vip.h5ad',
        'cell_type_Sst_Chodl.h5ad',
        'cell_type_Sst.h5ad',
        'cell_type_Pvalb.h5ad',
        'cell_type_Lamp5_Lhx6.h5ad',
        'cell_type_Sncg.h5ad')

# Loop over each cell type and run the task in parallel using background processes
for cell_type in "${cell_types[@]}"
do
    echo "Running task for cell type: ${cell_type}"
    python co-expression/CS-CORE/cs_core.py --data=data/split_train/"${cell_type}".h5ad &
done

# Wait for all background processes to finish
wait
