# Reference log-likelihoods for models with 1, 2 and 3 regimes from the R
# package MSTest (Rodriguez-Rondon and Dufour).
#
# Usage (from this folder):
#   Rscript generate_markov_number_of_regimes_reference.R
#
# Reads ms_two_regime_series.csv (a series with two recurring regimes that
# differ in mean, generated once in Python; see README.md) and writes
# ms_mstest_reference.json with, for the mean-switching model with a common
# variance and for the model in which the variance switches too:
#   - the log-likelihood of the model without switching (Nmdl) and of the
#     hidden Markov models with 2 and 3 regimes (HMmdl, EM algorithm from
#     several starting values, stationary initial distribution);
#   - the likelihood ratio statistics for 1 against 2 and 2 against 3
#     regimes;
#   - the estimates of the models with 2 and 3 regimes (entry [i, j] of a
#     transition matrix is the probability of moving from regime j to i).
# The series has no autoregressive terms, so the models are those of
# regimes' MarkovRegression with trend="c".

suppressMessages(library(MSTest))
library(jsonlite)

y <- as.matrix(read.csv("ms_two_regime_series.csv")$y)
set.seed(20261124)

one_model <- function(msvar) {
  m1 <- Nmdl(y, control = list(const = TRUE))
  fit <- function(k) {
    HMmdl(y, k = k, control = list(msmu = TRUE, msvar = msvar, use_diff_init = 30,
                                   maxit = 5000, thtol = 1e-10))
  }
  m2 <- fit(2)
  m3 <- fit(3)
  list(
    switching_variance = msvar,
    llf = list(k1 = m1$logLike, k2 = m2$logLike, k3 = m3$logLike),
    lr = list(k1_vs_k2 = 2 * (m2$logLike - m1$logLike),
              k2_vs_k3 = 2 * (m3$logLike - m2$logLike)),
    k2 = list(mu = as.numeric(m2$mu),
              sigma2 = sapply(m2$sigma, function(s) as.numeric(s)),
              transition = unname(m2$P)),
    k3 = list(mu = as.numeric(m3$mu),
              sigma2 = sapply(m3$sigma, function(s) as.numeric(s)),
              transition = unname(m3$P))
  )
}

meta <- list(
  generated = as.character(Sys.Date()),
  R = R.version.string,
  MSTest = as.character(packageVersion("MSTest"))
)
out <- list(meta = meta, nobs = nrow(y),
            models = list(one_model(FALSE), one_model(TRUE)))
write(toJSON(out, digits = 15, auto_unbox = TRUE, pretty = TRUE),
      "ms_mstest_reference.json")
cat("wrote ms_mstest_reference.json\n")
