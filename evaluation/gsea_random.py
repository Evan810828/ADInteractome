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

def plot_bubble_chart(results, output):
    """Show each cell type and method; colour is NES, size is -log10(p)."""
    from matplotlib.colors import TwoSlopeNorm

    if results.empty:
        raise ValueError("No GSEA results available for plotting")
    data = results.copy()
    data["NES"] = pd.to_numeric(data["NES"])
    data["p_value"] = pd.to_numeric(data["p_value"])
    if ((data["p_value"] <= 0) | (data["p_value"] > 1)).any():
        raise ValueError("p values must be in (0, 1]")
    cells = list(dict.fromkeys(data["cell_type"]))
    methods = list(dict.fromkeys(data["method"]))
    x = data["method"].map({v: i for i, v in enumerate(methods)})
    y = data["cell_type"].map({v: i for i, v in enumerate(cells)})
    sizes = 35 + 110 * -np.log10(data["p_value"])
    max_abs = max(float(np.abs(data["NES"]).max()), 1e-12)
    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 2), max(4, len(cells) * .4)))
    sc = ax.scatter(x, y, c=data["NES"], s=sizes, cmap="coolwarm",
                    norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
                    edgecolors="black", linewidths=.3)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_yticks(range(len(cells)), cells)
    ax.invert_yaxis()
    ax.set_title("BioGRID GSEA: NES and nominal p-value")
    fig.colorbar(sc, ax=ax, label="Normalized enrichment score (NES)")
    for p in (1, .05, .001):
        ax.scatter([], [], s=35 + 110 * -np.log10(p), color="gray", alpha=.6, label=f"p = {p:g}")
    ax.legend(title="Nominal p-value", bbox_to_anchor=(1.3, 1), loc="upper left")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BioGRID preranked GSEA")
    parser.add_argument("--data_path", default="evaluation/data")
    parser.add_argument("--input_dir", help="Rankings named METHOD__CELLTYPE.csv")
    parser.add_argument("--reference", help="Gene-set CSV (defaults to filtered_BioGRID.csv)")
    parser.add_argument("--output", default="res.csv", help="Combined results CSV")
    parser.add_argument("--bubble", help="Optional NES bubble chart PDF")
    parser.add_argument("--plot", action="store_true", help="Generate individual enrichment plots")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    input_dir = args.input_dir or os.path.join(args.data_path, "input")
    reference = args.reference or os.path.join(args.data_path, "filtered_BioGRID.csv")
    filenames = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    if not filenames:
        raise ValueError(f"No ranking CSV files in {input_dir}")
    gene_set = custom_gene_set(reference)
    results = []
    for filename in filenames:
        stem = os.path.splitext(filename)[0]
        if "__" not in stem:
            raise ValueError(f"Expected METHOD__CELLTYPE.csv, got {filename}")
        method, cell_type = stem.split("__", 1)
        if not method or not cell_type:
            raise ValueError(f"Expected METHOD__CELLTYPE.csv, got {filename}")
        res, pre_res = run_gsea(ranking_list(os.path.join(input_dir, filename)), gene_set)
        res.insert(0, "cell_type", cell_type)
        res.insert(0, "method", method)
        res.insert(2, "ranking_file", filename)
        results.append(res)
        if args.plot:
            plot_gsea(pre_res, os.path.join(args.data_path, "plots"), filename, args.top_n)
    combined = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_csv(args.output, index=False)
    if args.bubble:
        if combined["gene_set"].nunique() != 1:
            raise ValueError("Bubble chart requires exactly one reference gene set")
        os.makedirs(os.path.dirname(os.path.abspath(args.bubble)), exist_ok=True)
        plot_bubble_chart(combined, args.bubble)
    print(f"Wrote {len(combined)} GSEA results to {args.output}")


if __name__ == "__main__":
    main()
