"""
QuantGen Suite — CLI entry point
Usage:
    python main.py simulate   : simulate GWAS phenotype + genotype data
    python main.py gwas       : run GWAS (Python OLS per SNP)
    python main.py blup       : run BLUP / heritability via R (ASReml-style REML stub)
    python main.py plot       : generate Manhattan + QQ plots
    python main.py report     : full pipeline + HTML summary report
"""

import argparse
import sys
from pathlib import Path

from quantgen.simulate   import simulate_population
from quantgen.gwas       import run_gwas
from quantgen.plots      import manhattan_plot, qq_plot, genetic_trend_plot
from quantgen.heritability import reml_h2_python
from quantgen.r_bridge   import run_r_blup, r_available
from quantgen.report     import build_html_report


def cmd_simulate(args):
    print("=== Simulating population ===")
    data = simulate_population(
        n_animals=args.n,
        n_snps=args.snps,
        n_qtl=args.qtl,
        h2_true=args.h2,
        n_generations=args.gen,
        seed=args.seed,
    )
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    data["pheno"].to_csv(out / "phenotypes.csv", index=False)
    data["geno"].to_csv(out / "genotypes.csv", index=False)
    data["map"].to_csv(out / "snp_map.csv", index=False)
    data["pedigree"].to_csv(out / "pedigree.csv", index=False)
    print(f"  Saved {args.n} animals × {args.snps} SNPs → outputs/")
    print(f"  True h² = {args.h2},  QTL count = {args.qtl}")


def cmd_gwas(args):
    import pandas as pd
    print("=== Running GWAS (Python) ===")
    pheno = pd.read_csv("outputs/phenotypes.csv")
    geno  = pd.read_csv("outputs/genotypes.csv")
    snpmap = pd.read_csv("outputs/snp_map.csv")
    results = run_gwas(pheno, geno, snpmap, covariate_cols=["generation", "sex"])
    results.to_csv("outputs/gwas_results.csv", index=False)
    n_sig = (results["p_bonferroni"] < 0.05).sum()
    print(f"  Tested {len(results)} SNPs | Bonferroni-significant: {n_sig}")


def cmd_blup(args):
    import pandas as pd
    print("=== BLUP / heritability ===")
    pheno = pd.read_csv("outputs/phenotypes.csv")

    # Python REML (Henderson-style approximation)
    h2_py, sigma2_a, sigma2_e = reml_h2_python(pheno)
    print(f"  [Python] h² ≈ {h2_py:.3f}  σ²a = {sigma2_a:.3f}  σ²e = {sigma2_e:.3f}")

    # R BLUP if R is available
    if r_available():
        print("  [R] Running lme4 REML …")
        r_result = run_r_blup("outputs/phenotypes.csv", "outputs/pedigree.csv")
        print(f"  [R] h² ≈ {r_result['h2']:.3f}  σ²a = {r_result['sigma2_a']:.3f}"
              f"  σ²e = {r_result['sigma2_e']:.3f}")
    else:
        print("  [R] R not found on PATH — skipping R BLUP (Python result used)")


def cmd_plot(args):
    import pandas as pd
    print("=== Generating plots ===")
    Path("outputs").mkdir(exist_ok=True)
    gwas = pd.read_csv("outputs/gwas_results.csv")
    pheno = pd.read_csv("outputs/phenotypes.csv")
    manhattan_plot(gwas, out_path="outputs/manhattan.png")
    qq_plot(gwas["p_value"], out_path="outputs/qqplot.png")
    genetic_trend_plot(pheno, out_path="outputs/genetic_trend.png")
    print("  Saved: outputs/manhattan.png  qq_plot.png  genetic_trend.png")


def cmd_report(args):
    print("=== Full pipeline ===")
    cmd_simulate(args)
    cmd_gwas(args)
    cmd_blup(args)
    cmd_plot(args)
    print("=== Building HTML report ===")
    build_html_report("outputs/")
    print("  Report → outputs/report.html")


def main():
    p = argparse.ArgumentParser(description="QuantGen Suite — quantitative genetics toolkit")
    p.add_argument("command", choices=["simulate", "gwas", "blup", "plot", "report"])
    p.add_argument("--n",    type=int,   default=500,  help="Number of animals")
    p.add_argument("--snps", type=int,   default=5000, help="Number of SNPs")
    p.add_argument("--qtl",  type=int,   default=20,   help="Number of QTL")
    p.add_argument("--h2",   type=float, default=0.35, help="True heritability")
    p.add_argument("--gen",  type=int,   default=10,   help="Number of generations")
    p.add_argument("--seed", type=int,   default=42,   help="Random seed")
    args = p.parse_args()

    dispatch = {
        "simulate": cmd_simulate,
        "gwas":     cmd_gwas,
        "blup":     cmd_blup,
        "plot":     cmd_plot,
        "report":   cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
