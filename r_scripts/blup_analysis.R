# ============================================================
# r_scripts/blup_analysis.R
# QuantGen Suite — standalone R companion script
#
# This script mirrors the Python heritability.py analysis
# using R's native mixed-model ecosystem (lme4, ggplot2).
# It is called by Python via r_bridge.py, but can also be
# run interactively for verification or extension.
#
# Statistical model
# -----------------
# y_i = μ + b1·gen_i + b2·sex_i + a_i + e_i
#
# a_i ~ N(0, σ²_a)   (animal random effect)
# e_i ~ N(0, σ²_e)   (residual)
#
# REML estimate via lme4::lmer (Bates et al. 2015, J Stat Softw 67)
# ============================================================

suppressPackageStartupMessages({
  library(lme4)
  library(ggplot2)
  library(dplyr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
pheno_file  <- if (length(args) >= 1) args[1] else "outputs/phenotypes.csv"
result_file <- if (length(args) >= 2) args[2] else "outputs/r_blup_result.json"

# 1. Load data
pheno <- read.csv(pheno_file)
pheno$sex        <- factor(pheno$sex, labels = c("F", "M"))
pheno$animal_id  <- factor(pheno$animal_id)
pheno$generation <- as.numeric(pheno$generation)

cat("Animals:", nrow(pheno), "\n")
cat("Generations:", n_distinct(pheno$generation), "\n")

# 2. Fit mixed model by REML
fit <- lmer(phenotype ~ generation + sex + (1 | animal_id),
            data   = pheno,
            REML   = TRUE,
            control = lmerControl(optimizer = "bobyqa"))

# 3. Extract variance components
vc       <- as.data.frame(VarCorr(fit))
sigma2_a <- vc[vc$grp == "animal_id", "vcov"]
sigma2_e <- vc[vc$grp == "Residual",  "vcov"]
h2       <- sigma2_a / (sigma2_a + sigma2_e)
singular <- isSingular(fit)

cat(sprintf("\n--- REML results ---\n"))
cat(sprintf("  h²      = %.4f\n", h2))
cat(sprintf("  σ²_a    = %.4f\n", sigma2_a))
cat(sprintf("  σ²_e    = %.4f\n", sigma2_e))
cat(sprintf("  singular = %s\n", singular))

# 4. Extract EBVs (Best Linear Unbiased Predictors)
ebv <- ranef(fit)$animal_id
ebv$animal_id <- rownames(ebv)
colnames(ebv)[1] <- "ebv"
pheno_ebv <- left_join(pheno, ebv, by = "animal_id")

# 5. Genetic trend plot (ggplot2)
gen_means <- pheno_ebv %>%
  group_by(generation, sex) %>%
  summarise(mean_ebv = mean(ebv, na.rm = TRUE),
            se_ebv   = sd(ebv, na.rm = TRUE) / sqrt(n()),
            .groups = "drop")

p <- ggplot(gen_means, aes(x = generation, y = mean_ebv,
                            colour = sex, fill = sex)) +
  geom_ribbon(aes(ymin = mean_ebv - 1.96 * se_ebv,
                  ymax = mean_ebv + 1.96 * se_ebv), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  geom_smooth(method = "loess", se = FALSE, linetype = "dashed",
              linewidth = 0.7, show.legend = FALSE) +
  scale_colour_manual(values = c(F = "#009E73", M = "#0072B2"),
                      labels = c(F = "Female", M = "Male")) +
  scale_fill_manual(values   = c(F = "#009E73", M = "#0072B2"),
                    labels   = c(F = "Female", M = "Male")) +
  labs(x = "Generation", y = "Mean EBV",
       title = "Genetic trend (EBV) across selection generations",
       colour = NULL, fill = NULL) +
  theme_classic(base_size = 11) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold", size = 12))

ggsave("outputs/r_genetic_trend.png", p,
       width = 8, height = 4, dpi = 300, units = "in")
cat("  Saved: outputs/r_genetic_trend.png\n")

# 6. Write JSON result for Python handshake
result <- list(
  h2        = h2,
  sigma2_a  = sigma2_a,
  sigma2_e  = sigma2_e,
  converged = !singular
)
writeLines(toJSON(result, auto_unbox = TRUE), result_file)
cat(sprintf("  Result written to: %s\n", result_file))
