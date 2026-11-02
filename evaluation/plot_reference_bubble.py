"""Plot the 2026 reproduction alongside available reference baselines.

Lens and historical values are read from the table used for the supplied
reference figure; the 2026 reproduction values are read from gsea.csv.
These inputs have different data releases, splits and candidate universes.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CELL_ORDER = (
    "L6_IT", "Pvalb", "Lamp5_Lhx6", "Sst", "L5_ET", "Lamp5", "L2_3_IT", "Vip",
    "Sst_Chodl", "L6b", "Pax6", "L6_IT_Car3", "Chandelier", "L4_IT", "L6_CT",
    "L5_IT", "L5_6_NP", "Sncg",
)
METHODS = {
    "Lens-CelluFormer": ("Lens-CelluFormer (2026 test)", "#B07AA1"),
    "Lens-MLP": ("Lens-MLP (2026 test)", "#EDC948"),
    "CelluFormer": ("ADInteractome-CelluFormer (2026 train)", "#274A93"),
    "MLP": ("ADInteractome-MLP (2026 train)", "#1B7045"),
    "ADInteractome-CelluFormer": ("ADInteractome-CelluFormer (2024 historical)", "#4E79A7"),
    "ADInteractome-MLP": ("ADInteractome-MLP (2024 historical)", "#59A14F"),
    "cscore": ("CS-CORE (2024 historical)", "#F28E2B"),
    "pearson": ("Pearson (2024 historical)", "#E15759"),
    "cscore_2026": ("CS-CORE (2026 train)", "#A75B00"),
    "pearson_2026": ("Pearson (2026 train)", "#A1262F"),
}
REFERENCE_METHODS = tuple(k for k in METHODS if k not in ("CelluFormer", "MLP", "cscore_2026", "pearson_2026"))


def assemble(reference_path, reproduction_path, additional_path=None):
    ref = pd.read_csv(reference_path)
    new = pd.read_csv(reproduction_path)
    ref = ref.loc[ref["method"].isin(REFERENCE_METHODS)].copy()
    new = new.loc[new["method"].isin(("CelluFormer", "MLP"))].copy()
    if not (new["gene_set"] == "BioGRID").all() or not (ref["gene_set"] == "filtered_BioGRID").all():
        raise ValueError("Unexpected gene set")
    new["gene_percentage"] = pd.to_numeric(new["gene %"].astype(str).str.rstrip("%"))
    new["split"] = "train"
    new["status"] = "ok"
    new["gene_set"] = "filtered_BioGRID"
    columns = ["method", "cell_type", "split", "gene_set", "NES", "p_value", "gene_percentage", "status"]
    pieces = [ref[columns], new[columns]]
    active_methods = set(REFERENCE_METHODS) | {"CelluFormer", "MLP"}
    if additional_path is not None:
        additional = pd.read_csv(additional_path)
        additional["method"] = additional["method"].replace({"cscore": "cscore_2026", "pearson": "pearson_2026", "CS-CORE": "cscore_2026", "Pearson": "pearson_2026"})
        if not set(additional["method"]).issubset({"cscore_2026", "pearson_2026"}) or additional.empty:
            raise ValueError("Additional results must contain only 2026 CS-CORE/Pearson")
        if not additional["gene_set"].isin(("BioGRID", "filtered_BioGRID")).all():
            raise ValueError("Unexpected additional gene set")
        additional["gene_percentage"] = pd.to_numeric(additional["gene %"].astype(str).str.rstrip("%"))
        additional["split"], additional["status"], additional["gene_set"] = "train", "ok", "filtered_BioGRID"
        pieces.append(additional[columns])
        active_methods.update(additional["method"].unique())
    frame = pd.concat(pieces, ignore_index=True)
    expected = {(m, c) for m in active_methods for c in CELL_ORDER}
    observed = set(zip(frame["method"], frame["cell_type"]))
    if len(frame) != len(expected) or observed != expected or frame.duplicated(["method", "cell_type"]).any():
        raise ValueError(f"Unexpected method/cell pairs: missing={expected-observed}, extra={observed-expected}")
    for name in ("NES", "p_value", "gene_percentage"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    finite = frame["NES"].notna()
    if not np.isfinite(frame.loc[finite, ["NES", "p_value", "gene_percentage"]].to_numpy()).all():
        raise ValueError("Finite NES rows require finite p-values and gene percentages")
    if ((frame.loc[finite, "p_value"] <= 0) | (frame.loc[finite, "p_value"] > 1)).any():
        raise ValueError("p-values must be in (0, 1]")
    if ((frame.loc[finite, "gene_percentage"] < 0) | (frame.loc[finite, "gene_percentage"] > 100)).any():
        raise ValueError("Gene percentages must be in [0, 100]")
    frame["method_label"] = frame["method"].map(lambda x: METHODS[x][0])
    frame["source"] = np.where(frame["split"] == "train", "SEA-AD 2026 train; reproduction", np.where(
        frame["split"] == "test", "SEA-AD 2026 test; Lens reference", "SEA-AD 2024 historical"))
    return frame


def plot(frame, pdf_path):
    methods = [m for m in METHODS if m in set(frame["method"])]
    offsets = dict(zip(methods, np.linspace(-0.31, 0.31, len(methods))))
    y_lookup = {cell: i for i, cell in enumerate(CELL_ORDER)}

    def p_alpha(value):
        return 0.2 + 0.65 * np.clip(-np.log10(max(float(value), 1e-6)) / 6, 0, 1)

    with plt.rc_context({"pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(17.5, 11.5))
        for row in frame.itertuples(index=False):
            y = y_lookup[row.cell_type] + offsets[row.method]
            color = METHODS[row.method][1]
            if not np.isfinite(row.NES):
                ax.text(1.01, y, "NA", transform=ax.get_yaxis_transform(), color=color,
                        fontsize=8, va="center", clip_on=False)
                continue
            alpha = p_alpha(row.p_value)
            ax.hlines(y, 0, row.NES, color=color, alpha=max(alpha, .35), linewidth=.8, zorder=1)
            ax.scatter(row.NES, y, s=14*row.gene_percentage, color=color, edgecolors="black",
                       linewidths=1.8 if row.method.startswith("Lens-") else .3, alpha=alpha, zorder=2)
        ax.set_yticks(range(len(CELL_ORDER)), CELL_ORDER, fontsize=11)
        ax.set_ylim(len(CELL_ORDER)-.1, -.85)
        ax.set_xlabel("NES", fontsize=13)
        ax.set_title("GSEA analysis against filtered_BioGRID\n18 neuronal cell types: reproduction and baselines", fontsize=16, pad=14)
        ax.axvline(0, color="0.5", linewidth=.7, zorder=0)
        ax.grid(axis="x", linestyle="--", linewidth=.65, alpha=.35)
        ax.set_axisbelow(True)
        finite = frame.loc[np.isfinite(frame["NES"]), "NES"]
        lower, upper = min(0., float(finite.min())), max(0., float(finite.max()))
        margin = max((upper-lower)*.06, .12)
        ax.set_xlim(lower-margin, upper+margin)
        handles = [Patch(facecolor=METHODS[m][1], label=METHODS[m][0]) for m in methods]
        legend = ax.legend(handles=handles, title="Method", loc="upper left", bbox_to_anchor=(1.1, 1.0),
                           fontsize=9, title_fontsize=12, borderpad=.9, labelspacing=.75)
        ax.add_artist(legend)
        p_handles = [Line2D([], [], linestyle="", marker="o", markersize=9,
                            markerfacecolor="0.4", markeredgecolor="black", alpha=p_alpha(p),
                            label=f"p={p:g}") for p in (1e-6, .001, .1)]
        legend = ax.legend(handles=p_handles, title="p-value", loc="center left", bbox_to_anchor=(1.1, .43),
                           fontsize=10, title_fontsize=12)
        ax.add_artist(legend)
        percentage_handles = [Line2D([], [], linestyle="", marker="o", markersize=np.sqrt(14*v),
                                    markerfacecolor="white", markeredgecolor="black", label=f"{v}%") for v in (10, 30)]
        ax.legend(handles=percentage_handles, title="Gene Percentage", loc="lower left", bbox_to_anchor=(1.1, .08),
                  fontsize=10, title_fontsize=12, labelspacing=1.4, borderpad=1.0)
        fig.text(.12, .04,
                 "2026 train: new reproduction; CelluFormer checkpoint is epoch 2, batch 13000.\n"
                 "Lens: pair atoms only; 2026 test cells and expression-based candidates. Historical: archived 2024 results.\n"
                 "Data releases, splits and candidate universes differ; direct numerical comparisons require care. NA denotes unavailable NES.",
                 fontsize=8, ha="left", va="bottom", linespacing=1.5)
        fig.subplots_adjust(left=.12, right=.67, top=.91, bottom=.14)
        fig.savefig(pdf_path, metadata={"Title": "filtered_BioGRID enrichment: reproduction and baselines"})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--additional", type=Path, help="Optional GSEA CSV for 2026 CS-CORE/Pearson (18 rows per method)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    frame = assemble(args.reference, args.reproduction, args.additional)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    plot(frame, args.output)
    print(f"Wrote {len(frame)} rows to {args.table} and figure to {args.output}")


if __name__ == "__main__":
    main()
