"""
simulate.py — stochastic simulation of a selection experiment

Genetic model
-------------
Additive genetic value (breeding value) of animal i:

    a_i = Σ_j  α_j · x_ij       (sum over QTL loci j)

where x_ij ∈ {-1, 0, 1}  (centered genotype code)
and   α_j  ~ N(0, σ²_α)  is the allele-substitution effect at locus j.

Phenotype:
    y_i = μ + b·gen_i + a_i + e_i,    e_i ~ N(0, σ²_e)

Heritability (on observed scale):
    h² = σ²_a / (σ²_a + σ²_e)
       = σ²_a / σ²_p

We fix σ²_p = 1 and derive σ²_e = σ²_p - σ²_a.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PopConfig:
    n_animals: int      = 500
    n_snps: int         = 5_000
    n_qtl: int          = 20
    h2_true: float      = 0.35
    n_generations: int  = 10
    maf_min: float      = 0.05   # minimum minor-allele frequency
    selection_int: float = 1.0   # intensity (in σ units) applied to BV each gen
    sex_effect: float   = 2.5    # males heavier by this amount (e.g. BM42)
    seed: int           = 42


def simulate_population(
    n_animals: int      = 500,
    n_snps: int         = 5_000,
    n_qtl: int          = 20,
    h2_true: float      = 0.35,
    n_generations: int  = 10,
    seed: int           = 42,
) -> Dict[str, pd.DataFrame]:
    """Return dict with keys: pheno, geno, map, pedigree."""

    cfg = PopConfig(
        n_animals=n_animals, n_snps=n_snps, n_qtl=n_qtl,
        h2_true=h2_true, n_generations=n_generations, seed=seed
    )
    rng = np.random.default_rng(seed)
    return _simulate(cfg, rng)


# ── internal helpers ──────────────────────────────────────────────────────────

def _simulate(cfg: PopConfig, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n   = cfg.n_animals
    p   = cfg.n_snps
    q   = cfg.n_qtl
    G   = cfg.n_generations

    # 1. SNP map: assign to 20 autosomes, uniform positions
    chroms = rng.integers(1, 21, size=p)
    pos_mb = rng.uniform(0, 150, size=p)
    qtl_idx = rng.choice(p, size=q, replace=False)
    is_qtl = np.zeros(p, dtype=bool)
    is_qtl[qtl_idx] = True

    snp_map = pd.DataFrame({
        "snp_id": [f"SNP{i:05d}" for i in range(p)],
        "chr":    chroms,
        "pos_mb": pos_mb,
        "is_qtl": is_qtl,
    }).sort_values(["chr", "pos_mb"]).reset_index(drop=True)

    # Reorder qtl_idx to match sorted map
    qtl_idx_sorted = np.where(snp_map["is_qtl"].values)[0]

    # 2. QTL effects: α ~ N(0, σ²_α)
    #    σ²_a = 2 Σ p_j(1-p_j) α²_j  ≈ q · 2·0.25 · σ²_α  → fix σ²_a
    sigma2_a = cfg.h2_true          # phenotypic variance = 1 by convention
    sigma2_e = 1.0 - sigma2_a
    # Approximate σ²_α so that Var(a) ≈ h²
    sigma_alpha = np.sqrt(sigma2_a / (2 * 0.25 * q))
    qtl_effects = rng.normal(0, sigma_alpha, size=q)

    # 3. Generate genotypes + phenotypes per generation
    pheno_records  = []
    geno_records   = []
    pedigree_records = []
    animal_id = 0

    # Starting allele frequencies (MAF in [0.2, 0.5])
    freq = rng.uniform(0.2, 0.5, size=p)
    gen_effect_slope = 0.5  # phenotypic trend per generation (selection)

    for gen in range(G):
        n_gen = n // G
        for i in range(n_gen):
            animal_id += 1
            sex = rng.integers(0, 2)  # 0=female, 1=male

            # Genotype: 0/1/2 dosage, then center to -1/0/1
            dosage = rng.binomial(2, freq, size=p)
            x_centered = dosage - 2 * freq        # centered genotype

            # Breeding value from QTL only
            bv = x_centered[qtl_idx_sorted] @ qtl_effects

            # Phenotype
            mu  = 30.0 + cfg.sex_effect * sex      # baseline (mimics BM42)
            y   = mu + gen_effect_slope * gen + bv + rng.normal(0, np.sqrt(sigma2_e))

            pheno_records.append({
                "animal_id":  animal_id,
                "generation": gen,
                "sex":        sex,
                "phenotype":  round(y, 4),
                "true_bv":    round(bv, 4),
            })
            geno_records.append(dosage.tolist())
            pedigree_records.append({
                "animal_id": animal_id,
                "sire_id":   0,          # founder pedigree (simplified)
                "dam_id":    0,
                "generation": gen,
            })

        # Drift: small frequency shift each generation (simulate selection + drift)
        delta_p = 0.01 * rng.standard_normal(p)
        freq = np.clip(freq + delta_p, cfg.maf_min, 1 - cfg.maf_min)

    pheno_df   = pd.DataFrame(pheno_records)
    geno_df    = pd.DataFrame(geno_records, columns=snp_map["snp_id"].tolist())
    geno_df.insert(0, "animal_id", pheno_df["animal_id"].values)
    ped_df     = pd.DataFrame(pedigree_records)

    return {"pheno": pheno_df, "geno": geno_df, "map": snp_map, "pedigree": ped_df}
