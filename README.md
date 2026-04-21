# QuantGen Suite

A Python + R toolkit for quantitative genetics analysis, built around a
long-term directional selection experiment.  Demonstrates end-to-end competency
in both languages: Python for orchestration, data science, and statistics;
R for mixed-model REML and publication-quality ggplot2 figures.

---

## Capabilities

| Task | Language | Key library |
|------|----------|-------------|
| Population simulation (additive QTL model) | Python | NumPy |
| Single-SNP GWAS (OLS, Frisch–Waugh residualisation) | Python | SciPy, pandas |
| REML heritability (Henderson log-likelihood, L-BFGS-B) | Python | SciPy |
| REML heritability (lme4 REML, AI algorithm internally) | **R** | lme4 |
| Manhattan + QQ plots | Python | Matplotlib |
| Genetic trend + EBV plot | **R** | ggplot2 |
| HTML summary report | Python | — |
| Python → R subprocess bridge | Python | subprocess, json |

---

## Project structure

```
quantgen_suite/
├── main.py                   # CLI entry point
├── requirements.txt
├── outputs/                  # generated at runtime
│   ├── phenotypes.csv
│   ├── genotypes.csv
│   ├── gwas_results.csv
│   ├── manhattan.png
│   ├── qqplot.png
│   ├── genetic_trend.png
│   └── report.html           # self-contained HTML report
├── quantgen/
│   ├── simulate.py           # stochastic population + QTL simulation
│   ├── gwas.py               # GWAS engine + SimpleM threshold
│   ├── heritability.py       # REML via numerical optimisation
│   ├── plots.py              # Manhattan, QQ, genetic trend figures
│   ├── r_bridge.py           # Python → R subprocess interface
│   └── report.py             # HTML report generator
└── r_scripts/
    └── blup_analysis.R       # standalone R REML + ggplot2 trend figure
```

---

## Installation

```bash
pip install -r requirements.txt   # Python dependencies
# R dependencies (optional, for R BLUP):
#   install.packages(c("lme4", "ggplot2", "dplyr", "jsonlite"))
```

---

## Usage

```bash
# Full pipeline (simulate → GWAS → BLUP → plots → report)
python main.py report --n 600 --snps 5000 --qtl 20 --h2 0.35 --gen 12

# Individual steps
python main.py simulate --n 500 --snps 3000 --h2 0.40
python main.py gwas
python main.py blup
python main.py plot

# Standalone R script (run from project root)
Rscript r_scripts/blup_analysis.R outputs/phenotypes.csv
```

---

## Statistical background

### Genetic model

Additive breeding value of animal *i*:

```
a_i = Σ_j  α_j · x_ij
```

where `x_ij ∈ {-1, 0, 1}` is the centred dosage at QTL *j* and
`α_j ~ N(0, σ²_α)` is the allele-substitution effect.

Phenotypic model:

```
y_i = μ + b·gen_i + sex_i·δ + a_i + e_i,    e_i ~ N(0, σ²_e)
```

### REML heritability

The restricted log-likelihood (Henderson 1973):

```
ℓ_R(σ²_a, σ²_e) = -½ [log|V| + log|X'V⁻¹X| + y'Py]
```

where **V** = **ZAZ**'σ²_a + **I**σ²_e and
**P** = **V**⁻¹ − **V**⁻¹**X**(**X**'**V**⁻¹**X**)⁻¹**X**'**V**⁻¹.

Optimised over log σ²_a and log σ²_e (log-parameterisation enforces positivity).

### GWAS

Single-SNP OLS with Frisch–Waugh residualisation of fixed effects.
Genomic inflation: λ_GC = median(χ²_obs) / 0.4549.
Multiple testing: Bonferroni (α/m) and SimpleM effective tests.

---

## Python ↔ R bridge

`r_bridge.py` writes a templated R script to a temporary directory, calls
`Rscript --vanilla`, and reads the JSON result back into Python.
This pattern allows each language to do what it does best while keeping
the pipeline fully reproducible and language-agnostic at the interface.

```python
from quantgen.r_bridge import run_r_blup, r_available

if r_available():
    result = run_r_blup("outputs/phenotypes.csv", "outputs/pedigree.csv")
    print(result)  # {'h2': 0.312, 'sigma2_a': 1.73, 'sigma2_e': 3.81}
```

---

## Author

Cecilia Xi Ding — PhD researcher, quantitative genetics & genomics
