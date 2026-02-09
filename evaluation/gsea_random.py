#!/usr/bin/env python
# coding: utf-8

import gseapy as gp
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# Generate custom gene set
def custom_gene_set(file):
    df_gene_set = pd.read_csv(file)
    gene_set = df_gene_set.to_dict(orient='list')
    return gene_set

# Generate ranking list
def ranking_list(file):
    ranking = pd.read_csv(file)
    ranking = ranking.sort_values('avg_score', ascending=False)
    ranking = ranking.reset_index()
    ranking_sort = pd.DataFrame(np.sort(ranking[['gene1', 'gene2']], axis=1)) # sort gene_a and gene_b
    del ranking
    rnk = pd.DataFrame({'0':(ranking_sort[0]+','+ranking_sort[1]).tolist(),
                        '1':list(reversed(list(range(ranking_sort.shape[0]))))})
    rnk = rnk.set_index('0')
    return rnk

# Run GSEA
def run_gsea(rnk, gene_set):
    l_gene_set = []
    l_ES = []
    l_NES = []
    l_p_value = []
    l_tag = []
    l_gene = []
    pre_res = gp.prerank(rnk=rnk,
                         gene_sets=gene_set,
                         min_size=1,
                         max_size=1000000000,
                         permutation_num=1000, 
                         outdir=None,
                         seed=6,
                         verbose=True)
    terms = pre_res.res2d.Term
    for i in range(len(terms)):
        l_gene_set.append(terms[i])
        l_ES.append(pre_res.results[terms[i]]['es'])
        l_NES.append(pre_res.results[terms[i]]['nes'])
        if pre_res.results[terms[i]]['pval'] == 0:
            l_p_value.append(0.001) # min_p_value = 1/number of permutations
        else:
            l_p_value.append(pre_res.results[terms[i]]['pval'])   
        l_tag.append(pre_res.results[terms[i]]['tag %'])
        l_gene.append(pre_res.results[terms[i]]['gene %'])
    res_gsea = pd.DataFrame({'gene_set':l_gene_set,
                             'ES':l_ES,
                             'NES':l_NES,
                             'p_value':l_p_value,
                             'tag %':l_tag,
                             'gene %':l_gene})
    return res_gsea, pre_res

# Generate GSEA plots
def plot_gsea(pre_res, output_dir, file_name, top_n=5):
    """
    Generate GSEA enrichment plots for top significant gene sets
    
    Args:
        pre_res: prerank result object from gseapy
        output_dir: directory to save plots
        file_name: name of the file being processed
        top_n: number of top gene sets to plot (default: 5)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by NES and get top gene sets
    res_sorted = pre_res.res2d.sort_values('NES', ascending=False)
    top_terms = res_sorted.head(top_n)['Term'].tolist()
    
    # Plot each top term
    for term in top_terms:
        try:
            # Create a new figure for each plot
            fig, axes = plt.subplots(1, 1, figsize=(6, 5.5))
            
            # Create plot - gseaplot returns list of axes
            ax_list = gp.gseaplot(rank_metric=pre_res.ranking, 
                                  term=term, 
                                  **pre_res.results[term])
            
            # Save plot
            plot_name = f"{file_name.replace('.csv', '')}_{term.replace('/', '_').replace(' ', '_')}.png"
            plot_path = os.path.join(output_dir, plot_name)
            
            # Get the figure from the axes list
            if ax_list and len(ax_list) > 0:
                fig = ax_list[0].figure
            
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plot: {plot_path}")
        except Exception as e:
            print(f"Error plotting {term}: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')  # Clean up any open figures
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--data_path", type=str, default="evaluation/data", help="Path to the data folder")
    parser.add_argument("--output", type=str, default="res.csv", help="Path to the output file")
    parser.add_argument("--plot", action="store_true", help="Generate GSEA enrichment plots")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top gene sets to plot")
    args = parser.parse_args()
    
    for root, dirs, files in os.walk(os.path.join(args.data_path, "input")):
        for file in files:
            try:
                # Load data
                BioGRID = custom_gene_set(os.path.join(args.data_path, 'BioGRID.csv')) # Gene-Gene Interactions from BioGRID
                BioGRID_Strong = custom_gene_set(os.path.join(args.data_path,  'filtered_BioGRID.csv')) 
                DisGeNET_All = custom_gene_set(os.path.join(args.data_path, 'DisGeNET_GDA_0.csv')) # AD-related genes from DisGeNET (score_gda>0)
                DisGeNET_Strong = custom_gene_set(os.path.join(args.data_path,  'DisGeNET_GDA_01.csv')) # AD-related genes from DisGeNET (score_gda>0.1)
                rnk = ranking_list(os.path.join(args.data_path, "input", file)) # The ranking list of gene-gene interactions
                # GSEA
                res_gsea, pre_res = run_gsea(rnk, {**BioGRID_Strong})
                res_gsea.to_csv(os.path.join(args.data_path, "output", f"res_{file}"), index=False)
                
                # Generate plots if requested
                if args.plot:
                    plot_dir = os.path.join(args.data_path, "plots")
                    plot_gsea(pre_res, plot_dir, file, top_n=args.top_n)
                    
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
            