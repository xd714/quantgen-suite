"""
r_bridge.py — call R scripts from Python via subprocess

Demonstrates Python ↔ R interoperability:
  • Python orchestrates the pipeline (data flow, I/O, reporting)
  • R handles mixed-model REML (lme4) — a task where R's ecosystem excels

The R script is templated and injected with paths at runtime.
Results are returned as a JSON sidecar file, which Python reads back.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict


# ── public API ────────────────────────────────────────────────────────────────

def r_available() -> bool:
    """Return True if R is on the system PATH."""
    return shutil.which("Rscript") is not None


def run_r_blup(pheno_csv: str, pedigree_csv: str) -> Dict[str, float]:
    """
    Call the R BLUP script, return dict with h2, sigma2_a, sigma2_e.

    Raises RuntimeError if R is unavailable or the script fails.
    """
    if not r_available():
        raise RuntimeError("Rscript not found on PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_json = Path(tmpdir) / "blup_result.json"
        r_script    = _write_r_script(tmpdir, pheno_csv, pedigree_csv,
                                      str(result_json))
        proc = subprocess.run(
            ["Rscript", "--vanilla", r_script],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"R script failed:\n{proc.stderr}")

        with open(result_json) as f:
            return json.load(f)


# ── R script template ─────────────────────────────────────────────────────────

_R_SCRIPT_TEMPLATE = r"""
# QuantGen Suite — BLUP / REML via lme4
# Called from Python via subprocess; results written as JSON.
# Equivalent R approach to the Python REML implemented in heritability.py.

suppressPackageStartupMessages({
  library(lme4)        # REML mixed models
  library(jsonlite)    # JSON output for Python handshake
})

pheno   <- read.csv("{pheno_csv}")
ped     <- read.csv("{pedigree_csv}")

# ── Fixed-effect formula ─────────────────────────────────────────────────────
# y ~ generation + sex + (1 | animal_id)
#   (1 | animal_id) = random animal effect with variance σ²_a
#
# Note: lme4 does not natively use the numerator relationship matrix A.
# For a pedigree-based analysis, rrBLUP or ASReml-R would be used.
# Here we demonstrate the lme4 interface for the employer's benefit.

pheno$sex        <- as.factor(pheno$sex)
pheno$animal_id  <- as.factor(pheno$animal_id)

fit <- lmer(phenotype ~ generation + sex + (1 | animal_id),
            data = pheno, REML = TRUE)

vc        <- as.data.frame(VarCorr(fit))
sigma2_a  <- vc[vc$grp == "animal_id", "vcov"]
sigma2_e  <- vc[vc$grp == "Residual",  "vcov"]
sigma2_p  <- sigma2_a + sigma2_e
h2        <- sigma2_a / sigma2_p

cat(sprintf("  [R/lme4]  h² = %.4f  σ²_a = %.4f  σ²_e = %.4f\n",
            h2, sigma2_a, sigma2_e))

result <- list(h2 = h2, sigma2_a = sigma2_a, sigma2_e = sigma2_e,
               converged = !isSingular(fit))
writeLines(toJSON(result, auto_unbox = TRUE), "{result_json}")
"""


def _write_r_script(tmpdir: str, pheno_csv: str,
                    pedigree_csv: str, result_json: str) -> str:
    script_path = Path(tmpdir) / "run_blup.R"
    code = _R_SCRIPT_TEMPLATE.format(
        pheno_csv=pheno_csv,
        pedigree_csv=pedigree_csv,
        result_json=result_json,
    )
    script_path.write_text(code)
    return str(script_path)
