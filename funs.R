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
# vars <- list(xs = 'Petal.Length', ys = 'Petal.Width')
# fit_gradient_descent
#
fit_gradient_descent <- function(d,
                                 vars = list(ys, xs),
                                 starting_params = c(intercept = 0, slope = 0),
                                 learning_rate = 0.01,
                                 n_iter = 20) {
  xs <- d[[vars$xs]]
  ys <- d[[vars$ys]]

  np <- starting_params

  n <- length(ys)

  for (i in 1:n_iter) {
    preds <- np['intercept'] + np['slope'] * xs
    err   <- preds - ys

    gradient <- c(
      (2 / n) * sum(err),
      (2 / n) * sum(err * xs)
    )

    np <- np - learning_rate * gradient
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
# vars <- list(xs = 'Petal.Length', ys = 'Petal.Width')
# calculate_rmse(iris, vars, starting_params)
calculate_rmse <- function(d, vars = list(xs, ys), params) {
  predictions <- params['intercept'] + params['slope'] * d[[vars$xs]]
  sqrt(mean((d[[vars$ys]] - predictions)^2))
}


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

plot_org <- function(org, d, v = list(ys, xs), p, g = NULL, s = NULL, col = org + 1, l = TRUE) {
  xs <- d[[vars$xs]]
  ys <- d[[vars$ys]]

  plot(xs, ys,
       xlim = c(0, max(xs)),
       ylim = c(0, max(ys)),
       type = 'n', bty = 'l', las = 1,
       main = sprintf('Org %s: round %s', org, round_number))
  points(xs, ys, pch = 20)
  abline(g, col = 1, lwd = 3)
  abline(p, col = col, lwd = 3, lty = 2)
  abline(starting_params, col = 8, lwd = 2, lty = 3)
  # text(... params ...)
  if (l)
    legend('bottomright',
           legend = c('global', 'local', 'starting'),
           cex = .6, bty = 'n', lty = 1:3, col = c(1, col, 8))
}


sample_data <- function(d, n = nrow(d) %/% 2, p = nrow(d)) {
  d[sample(p, n, replace = TRUE), ]
}



print_params <- function(name, p, d = 3) {
  cat(
    name, '\n',
    'Petal.Length = ',
    round(p['intercept'], d),
    ' + ',
    round(p['slope'], d),
    ' * Petal.Width\n\n',
    sep = ''
  )

}
