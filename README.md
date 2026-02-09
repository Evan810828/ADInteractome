# Gene-Interact

Discovering gene-gene interactions from single-cell RNA-seq data using transformer attention mechanisms for Alzheimer's Disease (AD) classification.

## Overview

This project analyzes single-cell RNA-seq data from the [SEA-AD dataset](https://portal.brain-map.org/atlases-and-data/rnaseq/human-mtg-10x_sea-ad) (middle temporal gyrus region) to:

1. **Classify cells** as "Not AD" vs "High" AD neuropathological change using the CelluFormer model
2. **Discover gene-gene interactions** by extracting and analyzing attention maps from the trained transformer
3. **Evaluate interaction predictions** against known gene interaction databases (BioGRID, DisGeNET) using GSEA enrichment analysis
4. **Compare methods** including MLP baselines, co-expression methods (CS-CORE, Pearson correlation), and foundation models (scFoundation, scGPT, Geneformer, scMulan)

## Project Structure

```
gene-interact/
├── transformer/            # Core CelluFormer model
│   ├── celluformer.py      # CelluFormer architecture
│   ├── dataset.py          # VariableLengthSequenceDataset for sparse scRNA-seq data
│   ├── main.py             # Training and evaluation script
│   └── single_cell_tuning.py  # Cell-type-specific fine-tuning
├── mlp/                    # MLP baseline model
│   └── mlp.py              # MLP training for cell-type-specific classification
├── feature/                # Feature extraction and attention analysis
│   ├── attention_mean.py   # Extract attention-based gene interaction scores
│   ├── rank_attention.py   # Convert score maps to ranked gene-pair lists
│   ├── mlp_ranking.py      # MLP-based gene interaction ranking
│   └── pruning_test.py     # Attention pruning experiments
├── evaluation/             # Evaluation and enrichment analysis
│   └── gsea_random.py      # GSEA enrichment analysis against known gene databases
├── co-expression/          # Co-expression baseline methods
│   ├── CS-CORE/            # CS-CORE co-expression analysis
│   └── pearson_correlation/ # Pearson correlation analysis
├── foundation_models/      # Foundation model fine-tuning
│   ├── scFoundation/       # scFoundation fine-tuning and analysis
│   ├── geneformer/         # Geneformer fine-tuning
│   └── scMulan/            # scMulan fine-tuning
├── utils.py                # Shared utilities (setup_seed, load_checkpoint, etc.)
├── config.example.json     # Example configuration file
└── requirements.txt        # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Evan810828/gene-interact.git
cd gene-interact

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Download the SEA-AD MTG single-cell RNA-seq dataset:

```bash
mkdir -p data
wget -P data/ https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/RNAseq/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad
```

### Data Preprocessing

Split the dataset by cell type:

```bash
python data_split/SEA_AD.py
python data_split/split_cell_type.py
```

## Configuration

Copy the example config and modify as needed:

```bash
cp config.example.json config.json
```

Configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Training batch size | 128 |
| `learning_rate` | Learning rate | 1e-5 |
| `num_genes` | Number of genes in the dataset | 36601 |
| `embedding_dim` | Gene embedding dimension | 128 |
| `dim_feedforward` | Feedforward network dimension | 512 |
| `head` | Number of attention heads | 8 |
| `depth` | Number of transformer encoder layers | 4 |
| `dropout` | Dropout rate | 0.1 |

## Usage

### 1. Train the CelluFormer

```bash
python transformer/main.py \
    --config config.json \
    --data data/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad \
    --output output_models/global_training \
    --device_ids 0,1,2,3 \
    --epochs 4 \
    --eval_every 1000
```

### 2. Fine-tune on Specific Cell Types

```bash
python transformer/single_cell_tuning.py \
    --config config.json \
    --train_data data/split_train/cell_type_L5_ET.h5ad \
    --test_data data/split_test/cell_type_L5_ET.h5ad \
    --model_path output_models/global_training/<run_name>/epoch_2_batch_3473.pth \
    --output output_models/finetuning
```

### 3. Train MLP Baseline

```bash
python mlp/mlp.py --cell_type L5_ET --epochs 10
```

### 4. Extract Gene Interaction Scores

```bash
# Attention-based score extraction
python feature/attention_mean.py \
    --config config.json \
    --model_path <model_checkpoint>.pth \
    --data data/split_train/cell_type_L5_ET.h5ad \
    --output feature/score_maps/score_map_L5_ET.pkl

# Convert score maps to rankings
python feature/rank_attention.py --score_dir feature/score_maps/

# MLP-based rankings
python feature/mlp_ranking.py --cell_type L5_ET
```

### 5. GSEA Enrichment Analysis

```bash
python evaluation/gsea_random.py \
    --data_path evaluation/data \
    --output res.csv \
    --plot
```

## Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. Set your API key before training:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

## Cell Types

The analysis covers 18 cell types from the SEA-AD MTG dataset:

Chandelier, L2/3 IT, L4 IT, L5 ET, L5 IT, L5/6 NP, L6 IT, L6 CT, L6b, L6 IT Car3, Lamp5, Pax6, Vip, Sst Chodl, Sst, Pvalb, Lamp5 Lhx6, Sncg

## Citation

If you use this code in your research, please cite:

```
@misc{gene-interact,
    title={Gene-Interact: Discovering Gene-Gene Interactions via Transformer Attention},
    author={Evan},
    year={2024},
    url={https://github.com/Evan810828/gene-interact}
}
```
