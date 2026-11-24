# Reference p-values from the programs that accompany
# B. E. Hansen (1997), "Approximate Asymptotic P Values for Structural-Change
# Tests", Journal of Business and Economic Statistics 15, 60-67.
#
# Usage:
#   Rscript generate_hansen_pvalues.R <folder with pv_sup.R, pv_exp.R, pv_ave.R>
#
# The three files are in https://users.ssc.wisc.edu/~bhansen/progs/jbes_97r.zip
# (not redistributed here). Writes hansen1997_pvalues.csv in the current
# folder: one row per (statistic, value, q, pi0) with the p-value that
# Hansen's own function returns.

args <- commandArgs(trailingOnly = TRUE)
here <- getwd()
setwd(args[1])
source("pv_sup.R"); source("pv_exp.R"); source("pv_ave.R")
setwd(here)

rows <- c("kind,statistic,q,pi0,pvalue")
for (q in c(1, 2, 3, 4, 5, 10, 20, 40)) {
  for (pi0 in c(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.37, 0.485, 0.49)) {
    for (x in c(0.5, 2, 4, 6, 8.85, 12, 16)) {
      sup <- x + q
      rows <- c(rows, sprintf("sup,%.10f,%d,%.3f,%.15g", sup, q, pi0, pv_sup(sup, q, pi0)))
      ex <- sup / 3
      rows <- c(rows, sprintf("exp,%.10f,%d,%.3f,%.15g", ex, q, pi0, pv_exp(ex, q, pi0)))
      av <- sup / 2
      rows <- c(rows, sprintf("ave,%.10f,%d,%.3f,%.15g", av, q, pi0, pv_ave(av, q, pi0)))
    }
  }
}
writeLines(rows, "hansen1997_pvalues.csv")
cat("wrote hansen1997_pvalues.csv with", length(rows) - 1, "rows\n")
