#!/usr/bin/env Rscript
# =============================================================================
# generate_bai_perron_reference.R
#
# Generates reference values from R's strucchange package for cross-validation
# of the Python regimes.BaiPerronTest implementation.
#
# Usage:
#   Rscript tests/reference_data/generate_bai_perron_reference.R
#
# Output:
#   Prints break locations, SSR, BIC, and segment coefficients for three DGPs.
#   These values are hardcoded in:
#     tests/test_tests/test_bai_perron_reference.py
#
# Requirements:
#   install.packages("strucchange")
#
# References:
#   Bai, J., & Perron, P. (2003). Computation and analysis of multiple
#       structural change models. J. Applied Econometrics, 18(1), 1-22.
#   Zeileis, A., et al. (2002). strucchange: An R package for testing
#       for structural change. J. Statistical Software, 7(2), 1-38.
# =============================================================================

library(strucchange)

# Output directory — same folder as this script
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  out_dir <- dirname(script_path)
} else {
  out_dir <- "tests/reference_data"
}

cat("==============================================================================\n")
cat("Bai-Perron Reference Values from R strucchange\n")
cat("==============================================================================\n\n")

# ─────────────────────────────────────────────────────────────────────────────
# DGP 1: Simple mean shift (intercept-only model, one break)
#   y_t = 1 + e_t   for t = 1,...,100
#   y_t = 4 + e_t   for t = 101,...,200
# ─────────────────────────────────────────────────────────────────────────────
cat("--- DGP 1: Mean shift at t=100, n=200 ---\n")
set.seed(42)
y1 <- c(1 + rnorm(100, sd = 0.5), 4 + rnorm(100, sd = 0.5))
write.csv(data.frame(y = y1), file.path(out_dir, "dgp1_mean_shift.csv"),
          row.names = FALSE)

bp1 <- breakpoints(y1 ~ 1, h = 0.15, breaks = 3)
s1 <- summary(bp1)
cat("Break location(s):", bp1$breakpoints, "\n")
cat("RSS table:\n")
print(s1$RSS)
cat("Segment coefficients (1 break):\n")
print(coef(bp1, breaks = 1))
cat("\n")

# ─────────────────────────────────────────────────────────────────────────────
# DGP 2: Regression break (intercept + slope both break)
#   y_t = 1 + 0.5*x_t + e_t   for t = 1,...,100
#   y_t = 3 + 1.5*x_t + e_t   for t = 101,...,200
# ─────────────────────────────────────────────────────────────────────────────
cat("--- DGP 2: Regression break at t=100, n=200 ---\n")
set.seed(123)
n2 <- 200
x2 <- rnorm(n2)
y2 <- numeric(n2)
y2[1:100]   <- 1 + 0.5 * x2[1:100]   + rnorm(100, sd = 0.3)
y2[101:200] <- 3 + 1.5 * x2[101:200] + rnorm(100, sd = 0.3)
write.csv(data.frame(y = y2, x = x2), file.path(out_dir, "dgp2_regression_break.csv"),
          row.names = FALSE)

bp2 <- breakpoints(y2 ~ x2, h = 0.15, breaks = 3)
s2 <- summary(bp2)
cat("Break location(s):", bp2$breakpoints, "\n")
cat("RSS table:\n")
print(s2$RSS)
cat("Segment coefficients (1 break):\n")
print(coef(bp2, breaks = 1))
cat("\n")

# ─────────────────────────────────────────────────────────────────────────────
# DGP 3: Two breaks in mean (intercept-only)
#   y_t = 0 + e_t   for t = 1,...,80
#   y_t = 3 + e_t   for t = 81,...,160
#   y_t = 1 + e_t   for t = 161,...,240
# ─────────────────────────────────────────────────────────────────────────────
cat("--- DGP 3: Two mean shifts at t=80,160; n=240 ---\n")
set.seed(7)
y3 <- c(0 + rnorm(80, sd = 0.5), 3 + rnorm(80, sd = 0.5), 1 + rnorm(80, sd = 0.5))
write.csv(data.frame(y = y3), file.path(out_dir, "dgp3_two_breaks.csv"),
          row.names = FALSE)

bp3 <- breakpoints(y3 ~ 1, h = 0.15, breaks = 4)
s3 <- summary(bp3)
cat("Break location(s):", bp3$breakpoints, "\n")
cat("RSS table:\n")
print(s3$RSS)
cat("Segment coefficients (2 breaks):\n")
print(coef(bp3, breaks = 2))
cat("\n")

# ─────────────────────────────────────────────────────────────────────────────
# Save reference values as JSON-like text for easy parsing
# ─────────────────────────────────────────────────────────────────────────────
cat("=== MACHINE-READABLE REFERENCE VALUES ===\n")
cat(sprintf("DGP1_BREAKS=%s\n", paste(bp1$breakpoints, collapse=",")))
cat(sprintf("DGP1_RSS_0=%.10f\n", s1$RSS["RSS", "0"]))
cat(sprintf("DGP1_RSS_1=%.10f\n", s1$RSS["RSS", "1"]))
cat(sprintf("DGP1_RSS_2=%.10f\n", s1$RSS["RSS", "2"]))
cat(sprintf("DGP1_RSS_3=%.10f\n", s1$RSS["RSS", "3"]))

cat(sprintf("DGP2_BREAKS=%s\n", paste(bp2$breakpoints, collapse=",")))
cat(sprintf("DGP2_RSS_0=%.10f\n", s2$RSS["RSS", "0"]))
cat(sprintf("DGP2_RSS_1=%.10f\n", s2$RSS["RSS", "1"]))
cat(sprintf("DGP2_RSS_2=%.10f\n", s2$RSS["RSS", "2"]))
cat(sprintf("DGP2_RSS_3=%.10f\n", s2$RSS["RSS", "3"]))

cat(sprintf("DGP3_BREAKS=%s\n", paste(bp3$breakpoints, collapse=",")))
cat(sprintf("DGP3_RSS_0=%.10f\n", s3$RSS["RSS", "0"]))
cat(sprintf("DGP3_RSS_1=%.10f\n", s3$RSS["RSS", "1"]))
cat(sprintf("DGP3_RSS_2=%.10f\n", s3$RSS["RSS", "2"]))
cat(sprintf("DGP3_RSS_3=%.10f\n", s3$RSS["RSS", "3"]))
cat(sprintf("DGP3_RSS_4=%.10f\n", s3$RSS["RSS", "4"]))

cat("==============================================================================\n")
cat("CSV datasets saved to:", out_dir, "\n")
cat("Copy reference values into test_bai_perron_reference.py\n")
cat("==============================================================================\n")
