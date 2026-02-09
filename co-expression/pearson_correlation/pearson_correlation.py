import numpy as np
import scanpy as sc
import hdf5plugin
import pandas
import argparse
import pickle
import torch

neuron_list = [
        'cell_type_Chandelier.h5ad',
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
        'cell_type_Sncg.h5ad'
    ]


def read_data(data_path):
    data_cell = sc.read_h5ad(data_path)

    AD_cells = data_cell[data_cell.obs['Overall AD neuropathological Change'] == 'High']
        
    NonAD_cells = data_cell[data_cell.obs['Overall AD neuropathological Change'] == 'Not AD']
    
    return AD_cells.X.toarray(), NonAD_cells.X.toarray()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Pearson Correlation')
    # parser.add_argument('--data', type=str, help='Input file path')
    # args = parser.parse_args()
    
    for cell_type in neuron_list:
        input_path = f"data/split_train/{cell_type}"
        AD_data, NonAD_data = read_data(input_path)
        AD_pearson_corr = np.corrcoef(AD_data, rowvar=False)
        AD_pearson_corr = np.nan_to_num(AD_pearson_corr)
        AD_pearson_corr_tensor = torch.tensor(AD_pearson_corr)
        NonAD_pearson_corr = np.corrcoef(NonAD_data, rowvar=False)
        NonAD_pearson_corr = np.nan_to_num(NonAD_pearson_corr)
        NonAD_pearson_corr_tensor = torch.tensor(NonAD_pearson_corr)
        # score_diff_tensor = abs(AD_pearson_corr_tensor - NonAD_pearson_corr_tensor)
        score_diff_tensor = (AD_pearson_corr_tensor - NonAD_pearson_corr_tensor)
        
        # save as pkl
        cell_type = "_".join(input_path.split('/')[-1].split('.')[0].split('_')[2:])
        with open(f'co-expression/score_maps/{cell_type}_pearson_corr.pkl', 'wb') as f:
            pickle.dump(score_diff_tensor, f)