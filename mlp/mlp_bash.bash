# prompt input "cell_type" from user
cell_types=('Chandelier' 'L2_3_IT' 'L4_IT' 'L5_ET' 'L5_IT' 'L5_6_NP' 'L6_IT' 'L6_CT' 'L6b' 'L6_IT_Car3' 'Lamp5' 'Pax6' 'Vip' 'Sst_Chodl' 'Sst' 'Pvalb' 'Lamp5_Lhx6' 'Sncg')

for cell_type in "${cell_types[@]}"
do
    # if "output_models/MLP/L2_3_IT" path doesn't exists, start from training
    # else start from ranking
    if [ ! -d "output_models/MLP/$cell_type" ]; then
        echo "Training MLP for $cell_type"
        python mlp/mlp.py --cell_type $cell_type
    else
        python feature/mlp_ranking.py --cell_type $cell_type
    fi
done
wait