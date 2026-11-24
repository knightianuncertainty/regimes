# Reference values for the Andrews-Ploberger tests from the R package
# strucchange (Zeileis, Leisch, Hornik, Kleiber).
#
# Usage (from this folder):
#   Rscript generate_andrews_ploberger_reference.R
#
# Reads dgp1_mean_shift.csv and dgp2_regression_break.csv and writes
# ap_strucchange_reference.json. For each case it stores the whole sequence
# of F statistics of strucchange::Fstats (which are scaled by the number of
# coefficients, i.e. on the chi-squared scale), the sup, ave and exp
# statistics and the p-values of strucchange::sctest, which implement
# Hansen (1997).
#
# The cases "..._first_half" use the observations before the break only, so
# that the statistics are moderate: with a large break the exp statistic
# overflows in strucchange and its p-values are zero.

suppressMessages(library(strucchange))
library(jsonlite)

d1 <- read.csv("dgp1_mean_shift.csv")
d2 <- read.csv("dgp2_regression_break.csv")

one_case <- function(name, formula, data, from) {
  fs <- Fstats(formula, data = data, from = from)
  n <- nrow(data)
  out <- list(
    name = name,
    nobs = n,
    from = from,
    q = fs$nreg,
    # break fraction of the first and last statistic, times n: number of
    # observations in the first regime
    first_break = round(time(fs$Fstats)[1] * n),
    last_break = round(tail(time(fs$Fstats), 1) * n),
    f_sequence = as.numeric(fs$Fstats),
    breakpoint = fs$breakpoint
  )
  for (type in c("supF", "aveF", "expF")) {
    t <- sctest(fs, type = type)
    out[[paste0(type, "_statistic")]] <- unname(t$statistic)
    out[[paste0(type, "_pvalue")]] <- unname(t$p.value)
  }
  out
}

cases <- list(
  one_case("mean_shift_015", y ~ 1, d1, 0.15),
  one_case("mean_shift_010", y ~ 1, d1, 0.10),
  one_case("regression_break_015", y ~ x, d2, 0.15),
  one_case("mean_first_half_015", y ~ 1, d1[1:100, , drop = FALSE], 0.15),
  one_case("mean_first_half_005", y ~ 1, d1[1:100, , drop = FALSE], 0.05),
  one_case("regression_first_half_015", y ~ x, d2[1:100, ], 0.15),
  one_case("regression_first_half_020", y ~ x, d2[1:100, ], 0.20)
)

meta <- list(
  generated = as.character(Sys.Date()),
  R = R.version.string,
  strucchange = as.character(packageVersion("strucchange"))
)
write(toJSON(list(meta = meta, cases = cases), digits = 15, auto_unbox = TRUE, pretty = TRUE,
             na = "null"),
      "ap_strucchange_reference.json")
cat("wrote ap_strucchange_reference.json\n")
