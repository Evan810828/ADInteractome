# Gene-Interact

Discovering gene-gene interactions from single-cell RNA-seq data using transformer attention mechanisms for Alzheimer's Disease (AD) classification.

## Overview

This project analyzes single-cell RNA-seq data from the [SEA-AD dataset](https://portal.brain-map.org/atlases-and-data/rnaseq/human-mtg-10x_sea-ad) (middle temporal gyrus region) to:

1. **Classify cells** as "Not AD" vs "High" AD neuropathological change using the CelluFormer model
2. **Discover gene-gene interactions** by extracting and analyzing attention maps from the trained transformer
3. **Evaluate interaction predictions** against known gene interaction databases (BioGRID, DisGeNET) using GSEA enrichment analysis
4. **Compare methods** including MLP baselines, co-expression methods (CS-CORE, Pearson correlation, SHAP), and foundation models (scFoundation, scGPT, Geneformer)

## Project Structure

```
gene-interact/
├── transformer/                        # Core CelluFormer model
│   ├── celluformer.py                  # CelluFormer architecture
│   ├── dataset.py                      # VariableLengthSequenceDataset for sparse scRNA-seq
│   ├── main.py                         # Training and evaluation script
│   └── weighted_sampling_dataset.py    # Zero-inflated Consistent Weighted Sampling dataset
├── mlp/                                # MLP baseline model
│   ├── mlp.py                          # MLP training for cell-type-specific classification
│   └── mlp_bash.bash                   # Batch training/ranking across all cell types
├── feature/                            # Feature extraction and attention analysis
│   ├── attention_mean.py               # Extract attention-based gene interaction scores (full dataset)
│   ├── sampling_attention_mean.py      # Attention extraction with diversified/uniform sampling
│   ├── rank_attention.py               # Convert score maps to ranked gene-pair lists
│   ├── mlp_ranking.py                  # MLP-based gene interaction ranking
│   ├── pruning_test.py                 # Attention pruning validation experiments
│   ├── mlp_experiment.sh               # Batch MLP ranking for all cell types
│   ├── pruning_experiment.bash         # Batch pruning tests
│   ├── sampling_experiment.bash        # Batch sampling attention extraction
│   └── pruning/                        # Pruning gene pair files and results
├── evaluation/                         # Evaluation and enrichment analysis
│   ├── gsea_random.py                  # GSEA enrichment analysis against known gene databases
│   └── data/                           # BioGRID, DisGeNET reference datasets
├── co-expression/                      # Co-expression baseline methods
│   ├── CS-CORE/                        # CS-CORE co-expression analysis
│   │   ├── cs_core.py
│   │   └── cs_core.bash
│   ├── pearson_correlation/            # Pearson correlation analysis
│   │   ├── pearson_correlation.py
│   │   └── pearson_correlation.bash
│   └── shap/                           # Shapley interaction score analysis
│       ├── shape.py                    # SHAP interaction scores via MLP
│       └── deepshap.py                 # Deep SHAP-based GRN analysis
├── foundation_models/                  # Foundation model fine-tuning
│   ├── scFoundation/                   # scFoundation fine-tuning and attention extraction
│   │   ├── finetune.py
│   │   ├── average_attention_mean.py
│   │   ├── get_embedding.py
│   │   ├── load.py
│   │   ├── OS_scRNA_gene_index.19264.tsv
│   │   ├── models/
│   │   └── pretrainmodels/
│   ├── geneformer/                     # Geneformer fine-tuning
│   │   ├── finetune.py
│   │   ├── Geneformer_embedding_dict.csv
│   │   └── mart_export.tsv
│   └── scGPT/                          # scGPT fine-tuning and attention extraction
│       ├── finetune.py
│       ├── scGPT_attn_map.py
│       ├── scGPT_score_to_rank.py
│       └── gene_vocab.json
├── data_split/                         # Data preprocessing
│   ├── split_cell_type.py              # Split dataset by cell type
│   └── neuron_dataset.py               # Combine neuronal cell types
├── utils.py                            # Shared utilities (setup_seed, load_checkpoint, etc.)
└── requirements.txt                    # Python dependencies
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

Split the dataset by cell type (80/20 train/test, filtered to "Not AD" and "High"):

```bash
python data_split/split_cell_type.py \
    --data data/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad \
    --train_output_path data/split_train \
    --test_output_path data/split_test
```

Optionally, combine all neuronal cell types into a single file:

```bash
python data_split/neuron_dataset.py
```

## Configuration

Create a `config.json` file with the following parameters:

```json
{
    "num_genes": 36601,
    "embedding_dim": 128,
    "dim_feedforward": 512,
    "head": 8,
    "depth": 4,
    "dropout": 0.1,
    "batch_size": 128,
    "learning_rate": 1e-5
}
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_genes` | Number of genes in the dataset | 36601 |
| `embedding_dim` | Gene embedding dimension | 128 |
| `dim_feedforward` | Feedforward network dimension | 512 |
| `head` | Number of attention heads | 8 |
| `depth` | Number of transformer encoder layers | 4 |
| `dropout` | Dropout rate | 0.1 |
| `batch_size` | Training batch size | 128 |
| `learning_rate` | Learning rate | 1e-5 |

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

For evaluation only:

```bash
python transformer/main.py \
    --config config.json \
    --data data/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad \
    --model_path <checkpoint>.pth \
    --is_test
```

### 2. Train MLP Baseline

```bash
python mlp/mlp.py --cell_type L5_ET --epochs 10

# Or train all 18 cell types:
bash mlp/mlp_bash.bash
```

### 3. Extract Gene Interaction Scores

#### Attention-based (full dataset)

```bash
python feature/attention_mean.py \
    --config config.json \
    --model_path <model_checkpoint>.pth \
    --data data/split_train/cell_type_L5_ET.h5ad \
    --output feature/score_maps/score_map_L5_ET.pkl
```

#### Attention-based (with sampling)

```bash
python feature/sampling_attention_mean.py \
    --config config.json \
    --model_path <model_checkpoint>.pth \
    --data data/split_train/cell_type_L5_ET.h5ad \
    --sample_size 200 \
    --sample_mode diversified
```

#### Convert score maps to rankings

```bash
python feature/rank_attention.py --score_dir feature/score_maps/
```

#### MLP-based rankings

```bash
python feature/mlp_ranking.py --cell_type L5_ET

# Or all cell types:
bash feature/mlp_experiment.sh
```

### 4. Co-expression Baselines

#### CS-CORE

```bash
python co-expression/CS-CORE/cs_core.py \
    --data data/split_train/cell_type_L5_ET.h5ad

# Or all cell types:
bash co-expression/CS-CORE/cs_core.bash
```

#### Pearson Correlation

```bash
python co-expression/pearson_correlation/pearson_correlation.py
```

#### Shapley Interaction Scores

```bash
python co-expression/shap/shape.py --cell_type L5_ET
```

### 5. Foundation Model Fine-tuning

#### scFoundation

```bash
python foundation_models/scFoundation/finetune.py \
    --epochs 10 \
    --learning_rate 1e-4

# Extract attention-based scores:
python foundation_models/scFoundation/average_attention_mean.py \
    --config config.json \
    --model_path <scfoundation_checkpoint>.pth \
    --data data/split_train/cell_type_L5_ET.h5ad
```

#### scGPT

```bash
python foundation_models/scGPT/finetune.py

# Extract attention maps:
python foundation_models/scGPT/scGPT_attn_map.py \
    --data data/split_train/cell_type_L5_ET.h5ad

# Convert to rankings:
python foundation_models/scGPT/scGPT_score_to_rank.py
```

#### Geneformer

```bash
python foundation_models/geneformer/finetune.py \
    --epochs 10 \
    --batch_size 128
```

### 6. Pruning Validation

Validate discovered interactions by measuring accuracy impact when pruning specific gene pairs:

```bash
python feature/pruning_test.py \
    --model_path <model_checkpoint>.pth \
    --target_genes feature/pruning/gene_pair/L5_ET.txt \
    --config_path config.json \
    --mode pair

# Or batch:
bash feature/pruning_experiment.bash
```

### 7. GSEA Enrichment Analysis

Place ranking CSV files in `evaluation/data/input/`, then run:

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

WandB projects used:
- `gene-interact` — CelluFormer training
- `gene-interact-MLP` — MLP baseline training
- `scFoundation-finetune` — scFoundation fine-tuning
- `scGPT` — scGPT fine-tuning
- `gene-interact-Geneformer` — Geneformer fine-tuning

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
