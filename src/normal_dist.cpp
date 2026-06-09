#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "normal_dist.h"

arma::colvec cpp_mvrnorm_1(const arma::colvec & mu, const arma::mat & sigma) {
  int ncols = sigma.n_cols;
  arma::colvec Z = arma::randn(ncols);
  return mu + arma::chol(sigma).t() * Z;
}

double log_dmvn_both_mat_1_obs( const arma::colvec & x,
                                const arma::colvec & mu,
                                const arma::mat    & SigmaInv,
                                const arma::mat    & Sigma) {

  double d       = x.n_elem;
  arma::colvec z       = x - mu;
  double quad          = arma::as_scalar(z.t() * SigmaInv * z);
  double log_det_Sigma = arma::log_det_sympd(Sigma);
  return -0.5 * ( d * std::log(2.0 * arma::datum::pi) + log_det_Sigma + quad );
}

// [[Rcpp::export]]
double log_dmvn_mat_1_obs(const arma::colvec & x,
                          const arma::colvec & mu,
                          const arma::mat    & Sigma) {
  double d = x.n_elem;
  arma::colvec z = x - mu;
  double log_det_Sigma = arma::log_det_sympd(Sigma);
  arma::colvec sol = arma::solve( Sigma, z, arma::solve_opts::likely_sympd);
  double quad = arma::as_scalar(z.t() * sol);
  return -0.5 * (d * std::log(2.0 * arma::datum::pi) + log_det_Sigma + quad);
}

//' Sample from a multivariate Normal distribution
//'
//' Draws \code{n} independent samples from \eqn{\mathcal{N}(\mu, \Sigma)}
//' by computing the Cholesky factor of \eqn{\Sigma} once and applying it to
//' a \eqn{p \times n} matrix of standard normals.
//'
//' @param n Integer. Number of independent draws.
//' @param mu Numeric vector of length \eqn{p}: the mean.
//' @param sigma Symmetric positive-definite \eqn{p \times p} covariance matrix.
//'
//' @return A \eqn{p \times n} numeric matrix; column \code{i} is the i-th draw.
//'
//' @examples
//' p     <- 3
//' mu    <- c(1, -1, 2)
//' sigma <- diag(p)
//' X <- rmvnorm(n = 5000, mu = mu, sigma = sigma)
//' rowMeans(X)         # ~ mu
//' cov(t(X))           # ~ sigma
//'
//' @export
// [[Rcpp::export]]
arma::mat rmvnorm( const int n,
                   const arma::colvec & mu,
                   const arma::mat    & sigma) {
  const arma::uword p = sigma.n_rows;

  if (sigma.n_cols != p) Rcpp::stop("'sigma' must be a square matrix.");
  if (mu.n_elem != p)    Rcpp::stop("Length of 'mu' (%u) must match the dimension of 'sigma' (%u).", (unsigned) mu.n_elem, (unsigned) p);

  arma::mat L = arma::chol(sigma).t();    // L L' = sigma, triangolare inferiore
  arma::mat X = L * arma::randn(p, n);    // p x n, colonne ~ N(0, sigma)
  X.each_col() += mu;                     // broadcast della media su ogni colonna
  return X;
}
