import scanpy as sc
import hdf5plugin
import anndata
    

def split(type):
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
    
    data_cell = sc.read_h5ad("data/split_test/cell_type_L5_ET.h5ad")
    
    neuron_data = anndata.concat([sc.read_h5ad(f'data/split_{type}/{neuron}') for neuron in neuron_list])
    neuron_data.var = data_cell.var
    
    output_path = f'data/split_{type}/neuron_data.h5ad'
    print(f"Saving to {output_path}")
    neuron_data.write_h5ad(output_path, compression=hdf5plugin.FILTERS["zstd"], compression_opts=hdf5plugin.Zstd(clevel=5).filter_options)
    
    return neuron_data
     
if __name__ == '__main__':
    neuron_data_train = split('train')
    neuron_data_test = split('test')
    
    # store the whole neuron dataset
    neuron_data = anndata.concat([neuron_data_train, neuron_data_test])
    neuron_data.var = neuron_data_train.var
    neuron_data.write_h5ad('data/neuron_data.h5ad', compression=hdf5plugin.FILTERS["zstd"], compression_opts=hdf5plugin.Zstd(clevel=5).filter_options)