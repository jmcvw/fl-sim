LoadPackages(dplyr, tidyr, purrr, ggplot2)
# rm(new_global_params)
# i(reset = 0)

# ============================================================
# Simple Federated Learning simulation
# Bootstrap + rolling window + gradient descent
# ============================================================
#
# MODEL
#   Petal.Length ~ Petal.Width
#
# Three organizations; each organisation:
#   - has its own bootstrap sample of data (75 rows)
#   - receives 15 new observations per round
#   - retains only the most recent 75 observations
#       - (so each round 15 are droped from the start and a new 15 are added to the end)
#       - this simulates streamed data
#   - starts local training from the previous global parameters
#   - performs a limited number (default 20) of gradient-descent iterations
#   - sends its local parameters to the aggregator
#
# The aggregator:
#   - averages the local parameters
#   - produces new global parameters
#
# IMPORTANT:
#   The new global parameters become available for the
#   NEXT round. They are not used to alter the completed
#   round retrospectively.
#
# To simulate another round:
#   1. Change iter_number below
#   2. Copy the new global parameters into global_params
#   3. Run the script again
#
# ============================================================

# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------

# GRADIENT DESCENT
#
# We fit:
#
#   y = intercept + slope * x
#
# Unlike lm(), this does not jump directly to the optimum.
#
# Instead we start with the global parameters and take a
# limited number of small steps.
#
# This means the previous global parameters genuinely
# influence the local update.
#
fit_gradient_descent <- function(data,
                                 starting_params = c(intercept = 0, slope = 0),
                                 learning_rate = 0.1,
                                 n_iter = 20) {


  np <- starting_params

  n <- length(y)

  for (i in 1:n_iter) {
    preds <- np['intercept'] + np['slope'] * data$Petal.Width
    err   <- preds - data$Petal.Length

    np <- c(intercept, slope) - c((2 / n) * sum(err), (2 / n) * sum(err * x))
  }

  np
}

# RMSE FUNCTION
#
# RMSE is used here only to evaluate the parameters.
#
# It does NOT decide whether an organisation's parameters
# are accepted.
#
calculate_rmse <- function(data, params) {

  predictions <- params['intercept'] + params['slope'] * data[['Petal.Width']]

  sqrt(mean((data[['Petal.Length']] - predictions)^2))
}

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

plot_org <- function(org, d, p, g, col, l = TRUE) {
  plot(Petal.Length ~ Petal.Width, d,
       xlim = c(0, max(d$Petal.Width)),
       ylim = c(0, max(d$Petal.Length)),
       type = 'n', bty = 'l',
       main = sprintf('Org %s: round %s', org, iter_number))
  points(Petal.Length ~ Petal.Width, d, pch = 20)
  abline(g, col = 1, lwd = 3)
  abline(p, col = col, lwd = 3, lty = 2)
  # text(... params ...)
  if (l) legend('topleft', legend = c('global', 'local'), lty = 1:2, col = c(1, col))
}

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

n_per_org <- 75
new_rows_per_round <- 15
window_size <- n_per_org

learning_rate <- 0.01
n_iter <- 20

# ------------------------------------------------------------
# GLOBAL PARAMETERS
# ------------------------------------------------------------
#
# At the beginning there is no trained model, so we start
# with arbitrary parameters: interceopt = 0; slope = 0
#
# After each round, replace these values with the new aggregated global parameters

# rm(new_global_params)
global_params <- if (exists('new_global_params')) new_global_params else c(intercept = 3, slope = 0)

# ------------------------------------------------------------
# CREATE EACH ORGANISATION'S LOCAL DATA
# ------------------------------------------------------------
#
# Each organisation independently bootstraps 75 rows
# from iris, WITH replacement.
#
# The seed makes the exercise reproducible.

# i(reset = 0)
iter_number <- as.integer(i())

n_sample <- nrow(iris)

sample_data <- function(d, n = nrow(d) %/% 2, p = nrow(d)) {
  # sample_n(d, n, replace = TRUE)
  d[sample(p, n, replace = TRUE), ]
}

set.seed(99 + iter_number)
if (iter_number == 1) {
  org1_data <- sample_data(iris, n_sample)
  org2_data <- sample_data(iris, n_sample)
  org3_data <- sample_data(iris, n_sample)
} else {
  org1_data <- rbind(org1_data, sample_data(iris, 15))
  org2_data <- rbind(org2_data, sample_data(iris, 15))
  org3_data <- rbind(org2_data, sample_data(iris, 15))
}


# ------------------------------------------------------------
# SELECT THE DATA AVAILABLE THIS ROUND
# ------------------------------------------------------------
#
# Each organisation receives 15 new observations per round.

org1_window <- tail(org1_data, 75)
org2_window <- tail(org2_data, 75)
org3_window <- tail(org3_data, 75)


cat('\nROUND', iter_number, '\n')


# ------------------------------------------------------------
# EVALUATE THE PREVIOUS GLOBAL PARAMETERS
# ------------------------------------------------------------
#
# This tells us how the parameters received at the beginning
# of the round perform on each organisation's local window.

org1_rmse_before <- calculate_rmse(org1_window, global_params)
org2_rmse_before <- calculate_rmse(org2_window, global_params)
org3_rmse_before <- calculate_rmse(org3_window, global_params)


# ------------------------------------------------------------
# LOCAL TRAINING
# ------------------------------------------------------------
#
# Every organisation starts from the SAME global parameters.
#
# But each organisation performs the optimisation against
# its OWN local data.

org1_params <- fit_gradient_descent(org1_window, global_params)
org2_params <- fit_gradient_descent(org2_window, global_params)
org3_params <- fit_gradient_descent(org3_window, global_params)


# ------------------------------------------------------------
# EVALUATE LOCAL PARAMETERS
# ------------------------------------------------------------

org1_rmse_after <- calculate_rmse(org1_window, org1_params)
org2_rmse_after <- calculate_rmse(org2_window, org2_params)
org3_rmse_after <- calculate_rmse(org3_window, org3_params)

# ------------------------------------------------------------
# 8. DISPLAY LOCAL RESULTS
# ------------------------------------------------------------


results <- data.frame(
  org = 1:3,
  intercept = c(org1_params['intercept'], org2_params['intercept'], org3_params['intercept']),
  slope = c(org1_params['slope'], org2_params['slope'], org3_params['slope']),
  RMSE_before = c(org1_rmse_before, org2_rmse_before, org3_rmse_before),
  RMSE_after = c(org1_rmse_after, org2_rmse_after, org3_rmse_after)
)
results$improvement <- results$RMSE_before - results$RMSE_after

plot(results$org, ylim = range(results[4:5]), type = 'n', bty = 'o', las = 1)
segments(results$org, results$RMSE_before,
         results$org, results$RMSE_after,
         lwd = 6)
points(results$org, results$RMSE_before, pch = 19, col = 2, cex = 1.6)
points(results$org, results$RMSE_after, pch = 19, col = 4, cex = 1.6)

# ------------------------------------------------------------
# 9. AGGREGATION
# ------------------------------------------------------------
#
# The organisations send their LOCAL PARAMETERS.
#
# They do NOT send their data.
#
# For simplicity we use an unweighted mean.

local_params <- rbind(
  Org1 = org1_params,
  Org2 = org2_params,
  Org3 = org3_params
)

new_global_params <- colMeans(local_params)

op <- par(c('mfrow', 'mar'))
par(mar = c(2, 2, 2, 2), mfrow = c(2, 2))

# should have used dplyr + ggplot2!
plot_org(1, org1_data, org1_params, new_global_params, 2)
plot_org(2, org2_data, org2_params, new_global_params, 3)
plot_org(3, org3_data, org3_params, new_global_params, 4)

plot(Petal.Length ~ Petal.Width, iris, type = 'n', bty = 'l', main = paste('Population: round', iter_number))
points(Petal.Length ~ Petal.Width, org1_window, pch = 19, col = 2)
points(Petal.Length ~ Petal.Width, org2_window, pch = 20, col = 3)
points(Petal.Length ~ Petal.Width, org3_window, pch = 21, col = 4)
abline(org1_params, col = 2, lwd = 2)
abline(org2_params, col = 3, lwd = 2)
abline(org3_params, col = 4, lwd = 2)
abline(new_global_params, col = 1)
legend('topleft', legend = c('global', paste('org', 1:3)), lty = 1, col = 1:4)

par(op)
#
# lm1 <- lm(Petal.Length ~ Petal.Width, org1_data)$coeff
# plot(Petal.Length ~ Petal.Width, org1_data)
# abline(lm1, lwd = 3)
# abline(new_global_params, col = 2, lwd = 3)

# ------------------------------------------------------------
# DISPLAY THE NEW GLOBAL PARAMETERS
# ------------------------------------------------------------

new_global_params

print(results)

cat(
  'Petal.Length = ',
  round(new_global_params['intercept'], 3),
  ' + ',
  round(new_global_params['slope'], 3),
  ' * Petal.Width\n',
  sep = ''
)


# ------------------------------------------------------------
# END OF ROUND
# ------------------------------------------------------------
#
# IMPORTANT:
#
# new_global_params are the result of THIS round.
#
# They become the global parameters available at the
# START of the next round.
#
# To simulate another round:
#
#   1. Change:
#        iter_number <- 2
#
#   2. Change global_params to:
#        global_params <- new_global_params
#
#   3. Run the script again.
#
# The next round will:
#
#   - receive 15 additional observations
#   - move the rolling window (drop first 15 obs)
#   - start from the previous global parameters
#   - perform local gradient descent
#   - compare RMSE before/after
#   - aggregate the local parameters
#
# ============================================================

