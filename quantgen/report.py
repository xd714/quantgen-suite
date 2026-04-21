"""
report.py — generate a self-contained HTML summary report

Embeds all figures as base64 data-URIs (no external file dependencies).
Includes a summary statistics table from the GWAS results.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _img_b64(path: str) -> str:
    """Read a PNG and return a base64 data-URI."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


def build_html_report(outputs_dir: str = "outputs/") -> None:
    out = Path(outputs_dir)

    # Load data
    gwas  = pd.read_csv(out / "gwas_results.csv") if (out / "gwas_results.csv").exists() else pd.DataFrame()
    pheno = pd.read_csv(out / "phenotypes.csv")   if (out / "phenotypes.csv").exists()   else pd.DataFrame()

    # Summary stats
    n_snps   = len(gwas)
    n_sig    = int((gwas["p_bonferroni"] < 0.05).sum()) if "p_bonferroni" in gwas.columns else "—"
    lambda_gc = gwas["lambda_gc"].iloc[0] if "lambda_gc" in gwas.columns and len(gwas) > 0 else "—"
    n_animals = len(pheno)
    n_gen     = pheno["generation"].nunique() if "generation" in pheno.columns else "—"
    mean_pheno = round(pheno["phenotype"].mean(), 3) if "phenotype" in pheno.columns else "—"

    # Top hits table
    top_hits_html = ""
    if len(gwas):
        top = (gwas.sort_values("p_value").head(10)
                   [["snp_id", "chr", "pos_mb", "alpha_hat", "se", "p_value", "is_qtl"]]
                   .copy())
        top["p_value"]   = top["p_value"].map(lambda x: f"{x:.2e}")
        top["alpha_hat"] = top["alpha_hat"].map(lambda x: f"{x:.4f}")
        top["se"]        = top["se"].map(lambda x: f"{x:.4f}")
        top["is_qtl"]    = top["is_qtl"].map(lambda x: "✓" if x else "")
        top_hits_html = top.to_html(index=False, classes="data-table",
                                     border=0, justify="left")

    # Figures
    figs = {}
    for name, fname in [("manhattan", "manhattan.png"),
                         ("qqplot",    "qqplot.png"),
                         ("trend",     "genetic_trend.png")]:
        fp = out / fname
        figs[name] = _img_b64(str(fp)) if fp.exists() else None

    html = _render_html(
        n_animals=n_animals, n_gen=n_gen, n_snps=n_snps,
        n_sig=n_sig, lambda_gc=lambda_gc, mean_pheno=mean_pheno,
        top_hits_html=top_hits_html, figs=figs,
    )

    (out / "report.html").write_text(html, encoding="utf-8")


def _render_html(**ctx) -> str:
    fig_block = lambda key, title: (
        f'<div class="fig-block"><h3>{title}</h3>'
        f'<img src="{ctx["figs"][key]}" alt="{title}"></div>'
        if ctx["figs"].get(key) else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QuantGen Suite — Analysis Report</title>
<style>
  :root {{
    --bg: #f8f9fa; --surface: #ffffff; --primary: #0072B2;
    --text: #222; --muted: #666; --border: #dee2e6;
    --accent: #E69F00;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Helvetica Neue", Arial, sans-serif;
          background: var(--bg); color: var(--text); font-size: 14px; }}
  header {{ background: var(--primary); color: #fff; padding: 24px 40px; }}
  header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
  header p  {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px 20px; }}
  .card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 4px; }}
  .card .value {{ font-size: 24px; font-weight: 700; color: var(--primary); }}
  section {{ background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: 24px; margin-bottom: 28px; }}
  section h2 {{ font-size: 15px; font-weight: 700; margin-bottom: 16px;
               border-bottom: 2px solid var(--accent); padding-bottom: 6px; }}
  section h3 {{ font-size: 13px; font-weight: 600; margin-bottom: 10px;
               color: var(--muted); }}
  .fig-block {{ margin-bottom: 20px; }}
  .fig-block img {{ max-width: 100%; border: 1px solid var(--border);
                    border-radius: 4px; display: block; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.data-table th {{ background: var(--bg); border-bottom: 2px solid var(--border);
                         padding: 6px 10px; text-align: left; font-weight: 600; }}
  table.data-table td {{ padding: 5px 10px; border-bottom: 1px solid var(--border); }}
  table.data-table tr:hover {{ background: #f0f7ff; }}
  footer {{ text-align: center; padding: 20px; font-size: 11px; color: var(--muted); }}
  .badge {{ display: inline-block; background: var(--primary); color: #fff;
            font-size: 10px; border-radius: 3px; padding: 1px 6px; }}
</style>
</head>
<body>
<header>
  <h1>QuantGen Suite &mdash; Analysis Report</h1>
  <p>Single-trait GWAS &middot; REML heritability &middot; Genetic trend analysis
     &nbsp;&nbsp;<span class="badge">Python + R</span></p>
</header>
<main>
  <!-- Summary cards -->
  <div class="cards">
    <div class="card"><div class="label">Animals</div>
      <div class="value">{ctx["n_animals"]}</div></div>
    <div class="card"><div class="label">Generations</div>
      <div class="value">{ctx["n_gen"]}</div></div>
    <div class="card"><div class="label">SNPs tested</div>
      <div class="value">{ctx["n_snps"]}</div></div>
    <div class="card"><div class="label">Bonf. significant</div>
      <div class="value">{ctx["n_sig"]}</div></div>
    <div class="card"><div class="label">λ<sub>GC</sub></div>
      <div class="value">{ctx["lambda_gc"]}</div></div>
    <div class="card"><div class="label">Mean phenotype</div>
      <div class="value">{ctx["mean_pheno"]}</div></div>
  </div>

  <!-- Figures -->
  <section>
    <h2>GWAS Results</h2>
    {fig_block("manhattan", "Manhattan plot")}
    {fig_block("qqplot", "Quantile–Quantile (QQ) plot")}
  </section>

  <section>
    <h2>Genetic Trend</h2>
    {fig_block("trend", "Generation mean phenotype (female / male)")}
  </section>

  <!-- Top hits -->
  <section>
    <h2>Top 10 Associated SNPs</h2>
    <p style="font-size:12px;color:var(--muted);margin-bottom:12px;">
      ✓ = true simulated QTL &nbsp;|&nbsp;
      p-values shown before Bonferroni correction
    </p>
    {ctx["top_hits_html"]}
  </section>

  <!-- Methods note -->
  <section>
    <h2>Methods</h2>
    <p style="line-height:1.7;font-size:13px;">
      Genotypes and phenotypes were simulated under a purely additive infinitesimal
      model with <em>n</em><sub>QTL</sub> causal loci distributed randomly across
      20 autosomes. GWAS was conducted using single-SNP OLS regression with
      generation and sex as fixed covariates (Frisch–Waugh residualisation).
      Heritability was estimated by REML via numerical optimisation of the
      Henderson (1973) restricted log-likelihood, and independently by
      <code>lme4::lmer</code> in R. Multiple-testing correction used the
      genome-wide Bonferroni threshold. Genomic inflation (λ<sub>GC</sub>) was
      computed as median(χ²<sub>obs</sub>) / 0.4549.
    </p>
  </section>
</main>
<footer>
  Generated by <strong>QuantGen Suite v1.0</strong> &middot;
  Python (NumPy · SciPy · pandas · Matplotlib) + R (lme4)
</footer>
</body>
</html>
"""
