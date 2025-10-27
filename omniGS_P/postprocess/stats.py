# omniGS_P/postprocess/stats.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison


def compact_letter_display(tukey, alpha=0.05):
    """
    Generate Compact Letter Display (CLD) from Tukey HSD results.
    """
    groups = sorted(tukey.groupsunique)  # alphabetical order
    results = pd.DataFrame(
        data=tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    letters = {g: "" for g in groups}
    current_letter = "a"
    assigned = set()

    for g in groups:
        if g in assigned:
            continue
        letters[g] += current_letter
        assigned.add(g)

        # Add same letter to groups not significantly different from g
        for h in groups:
            if h in assigned:
                continue
            row = results[
                ((results["group1"] == g) & (results["group2"] == h)) |
                ((results["group1"] == h) & (results["group2"] == g))
            ]
            if not row.empty and row["reject"].values[0] is False:
                letters[h] += current_letter
                assigned.add(h)

        current_letter = chr(ord(current_letter) + 1)

    return letters


def run_anova_tukey(df, trait, metric, outdir):
    """
    Run one-way ANOVA and Tukey HSD for a given trait and metric.
    Save ANOVA table, Tukey results, and boxplot with significance letters.
    """
    os.makedirs(outdir, exist_ok=True)

    # ANOVA
    groups = [df[df["model"] == m][metric].values for m in df["model"].unique()]
    f_stat, p_val = f_oneway(*groups)

    anova_path = os.path.join(outdir, f"{trait}_anova.tsv")
    pd.DataFrame({
        "Trait": [trait],
        "Metric": [metric],
        "F_stat": [f_stat],
        "p_value": [p_val]
    }).to_csv(anova_path, sep="\t", index=False)

    # Tukey HSD
    mc = MultiComparison(df[metric], df["model"])
    tukey = mc.tukeyhsd(alpha=0.05)

    tukey_path = os.path.join(outdir, f"{trait}_tukey.tsv")
    pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]).to_csv(
        tukey_path, sep="\t", index=False
    )

    # CLD letters 
    letters_dict = compact_letter_display(tukey)

    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="model", y=metric, data=df,
        order=sorted(df["model"].unique()),   
        hue="model", palette="Set3", legend=False
    )
    
    
    y_min, y_max_all = df[metric].min(), df[metric].max()
    y_range = y_max_all - y_min
    
    # Add letters above each box in alphabetical order
    for pos, model in enumerate(sorted(df["model"].unique())):
        y_max = df[df["model"] == model][metric].max()
        plt.text(
            pos,
            y_max + 0.03 * y_range,   
            letters_dict.get(model, ""),
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )
    
    
    plt.ylim(y_min, y_max_all + 0.1 * y_range)
    
    plt.title(f"{trait}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()


    plot_path = os.path.join(outdir, f"{trait}_boxplot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()



def run_stats(geno_dir: str):
    """
    Perform ANOVA + Tukey test for each trait in geno_dir/phenotypes.
    Create anova/{pearsonr,r2} subfolders and save results.
    """
    phenotypes_dir = os.path.join(geno_dir, "phenotypes")
    if not os.path.exists(phenotypes_dir):
        raise FileNotFoundError(f"Phenotypes directory not found: {phenotypes_dir}")

    # Create output dirs
    anova_dir = os.path.join(geno_dir, "anova")
    pearsonr_dir = os.path.join(anova_dir, "pearsonr")
    r2_dir = os.path.join(anova_dir, "r2")
    os.makedirs(pearsonr_dir, exist_ok=True)
    os.makedirs(r2_dir, exist_ok=True)

    # Loop over traits
    for f in os.listdir(phenotypes_dir):
        if f.endswith(".tsv") and not f.endswith("_summary.tsv") and f != "summary_all_traits.tsv":
            trait_file = os.path.join(phenotypes_dir, f)
            trait = os.path.splitext(f)[0]

            df = pd.read_csv(trait_file, sep="\t")

            # Pearsonr
            run_anova_tukey(df, trait, "pearsonr", pearsonr_dir)

            # R2
            run_anova_tukey(df, trait, "model_r2", r2_dir)