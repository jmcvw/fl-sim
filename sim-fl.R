if (!exists('reset') || reset) {
  source('funs.R')
  reset <-  FALSE
  seed <- 99
  round_number <- 0
  vars <- list(ys = 'Petal.Length', xs = 'Petal.Width')
  d <- iris[unlist(vars)]
  # vars <- list(ys = '', xs = '')
  # d <- _____[unlist(vars)]
  starting_params <- aggregated_params <- c(intercept = 3, slope = 0)
}

# Settings

n_per_org <- nrow(d) %/% 2
new_rows_per_round <- nrow(d) %/% 10
window_size <- n_per_org

learning_rate <- 0.01
n_iter <- 20
n_rounds <- 4



for (round_number in seq_len(n_rounds)) {
  # round_number <- round_number + 1
  starting_params <- aggregated_params

  set.seed(seed + round_number)
  if (round_number == 1) {
    org1_data <- sample_data(d)
    org2_data <- sample_data(d)
    org3_data <- sample_data(d)
  } else {
    org1_data <- rbind(org1_data, sample_data(d, 15))
    org2_data <- rbind(org2_data, sample_data(d, 15))
    org3_data <- rbind(org2_data, sample_data(d, 15))
  }



  org1_window <- tail(org1_data, 75)
  org2_window <- tail(org2_data, 75)
  org3_window <- tail(org3_data, 75)


  org1_rmse_before <- calculate_rmse(org1_window, vars, starting_params)
  org2_rmse_before <- calculate_rmse(org2_window, vars, starting_params)
  org3_rmse_before <- calculate_rmse(org3_window, vars, starting_params)


  org1_params <- fit_gradient_descent(org1_window, vars, starting_params, learning_rate, n_iter)
  org2_params <- fit_gradient_descent(org2_window, vars, starting_params, learning_rate, n_iter)
  org3_params <- fit_gradient_descent(org3_window, vars, starting_params, learning_rate, n_iter)


  # Evaluate local parameters
  org1_rmse_after <- calculate_rmse(org1_window, vars, org1_params)
  org2_rmse_after <- calculate_rmse(org2_window, vars, org2_params)
  org3_rmse_after <- calculate_rmse(org3_window, vars, org3_params)



  results <- data.frame(
    org = 1:3,
    intercept = c(org1_params['intercept'], org2_params['intercept'], org3_params['intercept']),
    slope = c(org1_params['slope'], org2_params['slope'], org3_params['slope']),
    RMSE_before = c(org1_rmse_before, org2_rmse_before, org3_rmse_before),
    RMSE_after = c(org1_rmse_after, org2_rmse_after, org3_rmse_after)
  )
  results$improvement <- results$RMSE_before - results$RMSE_after


  # Display local results
  cat('\n## ROUND', round_number, '\n\n')

  cat('\n\n')
  cat('### Model improvement\n\n')

  plot(results$org, ylim = range(0, 3),
       type = 'n', bty = 'o', las = 1,
       xlab = 'Organization', ylab = 'RMSE',
       main = 'Change in RMSE',
       sub  = 'blue = before, red = after aggregation')

  segments(results$org, results$RMSE_before,
           results$org, results$RMSE_after,
           lwd = 6)
  points(results$org, results$RMSE_before, pch = 19, col = 4, cex = 1.6)
  points(results$org, results$RMSE_after, pch = 19, col = 2, cex = 1.6)



  # Aggregation
  local_params <- rbind(
    Org1 = org1_params,
    Org2 = org2_params,
    Org3 = org3_params
  )

  aggregated_params <- colMeans(local_params)

  op <- par(c('mfrow', 'mar'))
  par(mar = c(2, 2, 2, 2), mfrow = c(2, 2))


  cat('\n\n')
  cat('### Local models vs starting params\n\n')

  # should have used dplyr + ggplot2!
  plot_org(org = 1, tail(org1_data, 75), vars,
           org1_params, aggregated_params, starting_params)
  plot_org(org = 2, tail(org2_data, 75),
           vars, org2_params, aggregated_params, starting_params)
  plot_org(org = 3, tail(org3_data, 75),
           vars, org3_params, aggregated_params, starting_params)

  plot(Petal.Length ~ Petal.Width, d, type = 'n', bty = 'l', las = 1,
       main = paste('Population: round', round_number))
  points(Petal.Length ~ Petal.Width, d, pch = 19, col = 8, cex = 1)
  points(Petal.Length ~ jitter(Petal.Width, 1), org1_window, pch = 21, bg = 2)
  points(Petal.Length ~ jitter(Petal.Width, 1), org2_window, pch = 21, bg = 3)
  points(Petal.Length ~ jitter(Petal.Width, 1), org3_window, pch = 21, bg = 4)
  abline(org1_params, col = 2, lwd = 2)
  abline(org2_params, col = 3, lwd = 2)
  abline(org3_params, col = 4, lwd = 2)
  abline(aggregated_params, col = 1, lwd = 2)
  # legend('topleft', legend = c('global', paste('org', 1:3)), lty = 1, col = 1:4)
  par(op)


  # lm1 <- lm(Petal.Length ~ Petal.Width, org1_data)$coeff
  # plot(Petal.Length ~ Petal.Width, org1_data)
  # abline(lm1, lwd = 3)
  # abline(aggregated_params, col = 2, lwd = 3)


  # Display the new global parameters
  cat('\n\n')
  print_params('__Starting params__\n', starting_params)
  print_params('__Aggregated params__\n', aggregated_params)


  print(knitr::kable(round(results, 3), row.names = FALSE))

}
