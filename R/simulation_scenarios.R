#' Build an equicorrelated covariance matrix
#'
#' Builds a covariance matrix with given marginal variances and a single
#' equicorrelation \eqn{\rho} between every pair of components.
#'
#' @param sigma2 Numeric vector of length \eqn{q}: marginal variances.
#' @param rho Scalar in \eqn{[(q-1)^{-1}, 1]}: pairwise correlation.
#'
#' @return A \eqn{q \times q} symmetric positive-definite matrix.
#' @export
make_cov = function(sigma2, rho) {
  q       = length(sigma2)
  R       = matrix(rho, q, q)
  diag(R) = 1
  Dhalf   = diag(sqrt(sigma2))

  return(Dhalf %*% R %*% Dhalf)
}

#' Simulate data — Scenario 1 (random clusters from the centering measure \eqn{G_0})
#'
#' Three latent clusters (K = 3, sizes 50/100/50) drawn from a
#' Normal-Inverse-Wishart prior; two observed groups (J = 2). The
#' missingness pattern of cluster \eqn{k} is governed by a Bernoulli
#' template \eqn{c_k} (one per cluster), with
#' \eqn{\Pr(\text{observed} \mid c_{k,j} = 1) = 1 - \alpha} and
#' \eqn{\Pr(\text{observed} \mid c_{k,j} = 0) = \alpha}.
#'
#' @param seed Integer random seed.
#' @param alpha Probability of observing a "non-default" component
#'   (\eqn{c_{k,j} = 0}). Must lie in \eqn{[0, 1]}.
#' @param alpha0 Bernoulli probability used to draw the cluster templates
#'   \code{c_mat}. Must lie in \eqn{[0, 1]}.
#'
#' @return A list: \code{sim_pat} (group label, length n), \code{sim_clustr}
#'   (true cluster label 0..K-1, length n), \code{sim_truth} (n x q complete
#'   data), \code{sim_data} (n x q data with NAs), \code{RR_mat} (n x q
#'   observation indicators), \code{c_mat} (K x q), \code{Sigma_arr}
#'   (q x q x K), \code{mu_mat} (K x q).
#' @export
gen_data_S1 = function(seed, alpha) {
  set.seed(seed)

  ## ---- design ---------------------------------------------------------
  n = 200; q = 6; J = 2; K = 3
  n_j = rep(n / J, J)
  n_k = c(50, 100, 50)
  group    = rep(0:(J - 1),  times = n_j)  # 0-based group / pattern
  true_cls = rep(1:K,        times = n_k)  # 1-based cluster

  ## ---- NIW hyperparameters -------------------------------------------
  mu0 = rep(0, q);
  kappa0 = 1;
  Sigma0 = diag(q);
  nu0 = q + 2

  ## ---- cluster-level draws -------------------------------------------
  # Bernoulli "templates": c_mat[k, j] = 1 means component j is part of the
  # default observed pattern of cluster k.
  alpha0 = 0.4
  c_mat = sBNPi::rmvbern(n = K, p = rep(alpha0, q))                 # K x q

  # Joint draw of (mu_k, Sigma_k) for all clusters in a single call.
  niw       = sBNPi::rniw(n = K, mean = mu0, Psi = Sigma0, kappa = kappa0, nu = nu0)
  mu_mat    = t(niw$mu)                                                # K x q
  Sigma_arr = niw$Sigma                                                # q x q x K

  ## ---- subject-level draws -------------------------------------------
  Y_mat_complete = matrix(NA, n, q)
  RR_mat         = matrix(NA, n, q)

  for (k in seq_len(K)) {
    I_k = which(true_cls == k)

    # P(observed): alpha where c = 0, (1 - alpha) where c = 1.
    p_c_k         = alpha * (c_mat[k, ] == 0) + (1 - alpha) * (c_mat[k, ] == 1)
    RR_mat[I_k, ] = sBNPi::rmvbern(n = length(I_k), p = p_c_k)

    # rmvnorm returns p x n_k; transpose to put subjects on rows.
    Y_mat_complete[I_k, ] = t(sBNPi::rmvnorm(n     = length(I_k),
                                             mu    = mu_mat[k, ],
                                             sigma = Sigma_arr[, , k]))
  }

  # Apply the missingness mask.
  Y_mat_obs               = Y_mat_complete
  Y_mat_obs[RR_mat == 0]  = NA

  list(sim_pat    = group,
       sim_clustr = true_cls - 1,
       sim_truth  = Y_mat_complete,
       sim_data   = Y_mat_obs,
       RR_mat     = RR_mat,
       c_mat      = c_mat,
       Sigma_arr  = Sigma_arr,
       mu_mat     = mu_mat)
}

#' Simulate data — Scenario 2 (fixed atoms, equicorrelated covariance)
#'
#' Three fixed atoms (K = 3, sizes 50/100/50). Means and templates are
#' deterministic; covariances are equicorrelated (variance 2 on every
#' component, pairwise correlation \code{rho}). Missingness follows the
#' same parameterisation as \code{\link{gen_data_S1}}.
#'
#' @param alpha Probability of observing a non-default component
#'   (\eqn{c = 0}); see \code{\link{gen_data_S1}}.
#' @param rho Pairwise correlation used in \code{\link{make_cov}}.
#' @param seed Integer random seed.
#' @return Same list structure as \code{\link{gen_data_S1}}.
#' @export
gen_data_S2 = function(seed, alpha, rho) {
  set.seed(seed)

  n = 200; q = 6; J = 2; K = 3
  n_k = c(50, 100, 50)
  group    = rep(0:(J - 1), each = n / J)
  true_cls = rep(1:K, times = n_k)

  # Templates (S1 convention).
  c_mat = rbind(rep(1, q),                        # cluster 1: never censored
                 c(rep(0, 2), rep(1, q - 2)),     # cluster 2: first 2 censored
                 c(rep(1, 2), rep(0, q - 2)))     # cluster 3: last 4 censored

  mu_mat = rbind(rep(-3, q), rep(0, q), rep(3, q))

  # Equicorrelated covariance, same variance (= 2) for every cluster/component.
  sigma2_mat = matrix(2, K, q)
  Sigma_arr  = array(NA, c(q, q, K))
  for (k in seq_len(K)) Sigma_arr[, , k] = make_cov(sigma2_mat[k, ], rho)

  Y_mat_complete = matrix(NA, n, q)
  RR_mat         = matrix(NA, n, q)

  for (k in seq_len(K)) {
    I_k = which(true_cls == k)
    # Same missingness logic as Scenario 1.
    p_c_k         = alpha * (c_mat[k, ] == 0) + (1 - alpha) * (c_mat[k, ] == 1)
    RR_mat[I_k, ] = sBNPi::rmvbern(n = length(I_k), p = p_c_k)
    Y_mat_complete[I_k, ] = t(sBNPi::rmvnorm(n     = length(I_k),
                                             mu    = mu_mat[k, ],
                                             sigma = Sigma_arr[, , k]))
  }

  Y_mat_obs              = Y_mat_complete
  Y_mat_obs[RR_mat == 0] = NA

  list(sim_pat    = group,
       sim_clustr = true_cls - 1,
       sim_truth  = Y_mat_complete,
       sim_data   = Y_mat_obs,
       RR_mat     = RR_mat,
       c_mat      = c_mat,
       Sigma_arr  = Sigma_arr,
       mu_mat     = mu_mat)
}
# CHECKED UP TO HERE (1 & 2)

#' Simulate data — Scenario 3 (fixed atoms, structured MAR)
#'
#' Four fixed atoms (K = 3, sizes 50/100/50) on two groups (J = 2). Cluster
#' means and covariances are fixed; missingness depends on the cluster
#' through a fixed template \code{c_mat}: each component with
#' \eqn{c_{k,j} = 0} is independently missing with probability
#' \code{CensProb}, components with \eqn{c_{k,j} = 1} are always observed.
#' The reference pattern is \eqn{(0,0,0,1,1,1)^\top} for the first cluster,
#' \eqn{(1,1,1,1,1,1)^\top} for the second (shared between groups), and \eqn{(1,1,1,0,0,0)^\top}
#' for the third.
#'
#' @param CensProb Probability of missingness on the components that are
#'   subject to censoring.
#' @param sd Marginal standard deviation common to all components.
#' @param seed Integer random seed.
#' @return Same list structure as \code{\link{gen_data_S1}}.
#' @export
gen_data_S3 = function(CensProb, sd, seed) {
  set.seed(seed)

  n = 200; q = 6; J = 2; K = 4 # K = 3, 4 is for code
  n_k = rep(50, K)
  group    = rep(0:(J - 1), each = n / J)
  true_cls = rep(1:K, times = n_k)

  ## Two correlation structures shared across atoms.
  R_pos = matrix(0.5,  q, q); diag(R_pos) = 1
  R_neg = matrix(-1/7, q, q); diag(R_neg) = 1
  S_pos = sd^2 * R_pos
  S_neg = sd^2 * R_neg

  ## Atoms g00, g01, g10, g11
  mu_mat = rbind( c( 3,  3,  3,  3,  3,  3),    # g00
                  c( 0,  0,  0,  0,  0,  0),    # g01
                  c( 0,  0,  0,  0,  0,  0),    # g10
                  c(-3, -3, -3, -3, -3, -3))    # g11

  Sigma_arr = array(NA, c(q, q, K))
  Sigma_arr[, , 1] = S_neg
  Sigma_arr[, , 2] = S_pos
  Sigma_arr[, , 3] = S_pos
  Sigma_arr[, , 4] = S_neg

  # Templates (S1 convention: 1 = default observed, 0 = MAR-censored).
  c_mat = rbind( c(0, 0, 0, 1, 1, 1),    # g00: missing only on first 3
                 c(0, 0, 0, 0, 0, 0),    # g01: missing on all 6
                 c(0, 0, 0, 0, 0, 0),    # g10: missing on all 6
                 c(1, 1, 1, 0, 0, 0))    # g11: missing only on last 3

  Y_mat_complete = matrix(NA, n, q)
  RR_mat         = matrix(NA, n, q)

  for (k in seq_len(K)) {
    I_k = which(true_cls == k)

    # P(observed) = (1 - CensProb) where c = 0, exactly 1 where c = 1.
    p_c_k         = (1 - CensProb) * (c_mat[k, ] == 0) + 1 * (c_mat[k, ] == 1)
    RR_mat[I_k, ] = sBNPimp::rmvbern(n = length(I_k), p = p_c_k)

    Y_mat_complete[I_k, ] = t(sBNPimp::rmvnorm( n     = length(I_k),
                                                mu    = mu_mat[k, ],
                                                sigma = Sigma_arr[, , k]))
  }

  Y_mat_obs              = Y_mat_complete
  Y_mat_obs[RR_mat == 0] = NA

  list(sim_pat    = group,
       sim_clustr = true_cls - 1,
       sim_truth  = Y_mat_complete,
       sim_data   = Y_mat_obs,
       RR_mat     = RR_mat,
       c_mat      = c_mat,
       Sigma_arr  = Sigma_arr,
       mu_mat     = mu_mat)
}

#' Simulate data — Scenario 4 (fixed atoms, MCAR missingness)
#'
#' Same atoms as \code{\link{gen_data_S3}}, but missingness is completely
#' at random: every entry is independently missing with probability
#' \code{CensProb}, regardless of cluster or component.
#'
#' @inheritParams gen_data_S3
#' @return Same list structure as \code{\link{gen_data_S1}}.
#' @export
gen_data_S4 = function(seed, CensProb, sd) {
  set.seed(seed)

  n = 200; q = 6; J = 2; K = 4
  n_k = rep(50, K)
  group    = rep(0:(J - 1), each = n / J)
  true_cls = rep(1:K, times = n_k)

  R_pos = matrix(0.5,  q, q); diag(R_pos) = 1
  R_neg = matrix(-1/7, q, q); diag(R_neg) = 1
  S_pos = sd^2 * R_pos
  S_neg = sd^2 * R_neg

  mu_mat = rbind(c( 3,  3,  3,  3,  3,  3),
                  c(-3, -3, -3,  3,  3,  3),
                  c(-3, -3, -3,  3,  3,  3),
                  c(-3, -3, -3, -3, -3, -3))

  Sigma_arr = array(NA, c(q, q, K))
  Sigma_arr[, , 1] = S_neg
  Sigma_arr[, , 2] = S_pos
  Sigma_arr[, , 3] = S_pos
  Sigma_arr[, , 4] = S_neg

  # MCAR: no informative template -> all zeros.
  c_mat = matrix(0, K, q)

  Y_mat_complete = matrix(NA, n, q)
  RR_mat         = matrix(NA, n, q)

  for (k in seq_len(K)) {
    I_k = which(true_cls == k)
    p_c_k = (1 - CensProb) * (c_mat[k, ] == 0) + 1 * (c_mat[k, ] == 1)
    RR_mat[I_k, ] = sBNPimp::rmvbern(n = length(I_k), p = p_c_k)
    Y_mat_complete[I_k, ] = t(sBNPimp::rmvnorm(n     = length(I_k),
                                                mu    = mu_mat[k, ],
                                                sigma = Sigma_arr[, , k]))
  }

  Y_mat_obs              = Y_mat_complete
  Y_mat_obs[RR_mat == 0] = NA

  list(sim_pat    = group,
       sim_clustr = true_cls - 1,
       sim_truth  = Y_mat_complete,
       sim_data   = Y_mat_obs,
       RR_mat     = RR_mat,
       c_mat      = c_mat,
       Sigma_arr  = Sigma_arr,
       mu_mat     = mu_mat)
}

