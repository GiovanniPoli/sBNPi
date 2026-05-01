#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <tuple>

#include "normal_inverse_wishart_dist.h"
#include "normal_dist.h"

arma::mat cpp_rwish_1( const double nu, const arma::mat & S) {
  int p = S.n_rows ;
  arma::mat Schol = arma::chol(S) ;
  arma::mat Z     = arma::randn(nu, p) * Schol ;
  return Z.t()*Z ;
}

std::tuple<arma::colvec, arma::mat, arma::mat> rniw_cpp( const arma::colvec & ex,
                                                         const arma::mat    & PsiInv,
                                                         const double kappa,
                                                         const double nu ){
  // Sample Sigma ~ Inverse-Wishart
  arma::mat SigmaInv = cpp_rwish_1(nu, PsiInv);        // SigmaInv ~ Wishart(Psi^-1, nu)
  arma::mat Sigma    = arma::inv_sympd(SigmaInv);      // Sigma ~ Inv-Wishart(Psi, nu)
  arma::colvec    mu = cpp_mvrnorm_1(ex, Sigma / kappa);

  std::tuple<arma::colvec, arma::mat, arma::mat> ret = std::make_tuple(mu, SigmaInv, Sigma);

  return ret;
}

std::tuple<arma::colvec, double, double, arma::mat> post_param_niw( const  arma::mat    & X,
                                                                    const  arma::colvec & m0,
                                                                    const  arma::mat    & Psi0,
                                                                    double kappa0,
                                                                    double nu0 ){
  // Note: To be coherent with the MCMC input, for n=1 X must be a matrix with 1 row (ok)
  int n = X.n_rows;
  arma::colvec x_bar = arma::mean(X, 0).t();
  arma::mat diff = X.each_row() - x_bar.t();
  arma::mat S = diff.t() * diff;

  double kappa_n = kappa0 + n;
  arma::vec m_n = (kappa0 * m0 + n * x_bar) / kappa_n;
  double nu_n = nu0 + n;
  arma::mat Sigma_n = Psi0 + S + (kappa0 * n / kappa_n) * (x_bar - m0) * (x_bar - m0).t();

  Sigma_n = arma::symmatu(Sigma_n); // enforce symmetry
  return std::make_tuple(m_n, kappa_n, nu_n, Sigma_n);
}

//' Sample from a Wishart distribution
//'
//' Generates \code{n} independent draws from a Wishart distribution with
//' \code{nu} degrees of freedom and scale matrix \code{S}.
//'
//' @param n Integer. Number of independent matrices to draw.
//' @param nu Degrees of freedom. Must satisfy \code{nu >= nrow(S)} for the
//'   distribution to be non-degenerate (the function does not check this).
//' @param S Symmetric positive-definite scale matrix of dimension
//'   \eqn{p \times p}.
//'
//' @return A numeric array of dimension \eqn{p \times p \times n}; the
//'   \code{i}-th slice \code{[, , i]} is the \code{i}-th Wishart draw.
//'
//' @examples
//' S  <- diag(3)
//' nu <- 10
//' draws <- rwish(n = 5000, nu = nu, S = S)
//' dim(draws)                     # 3 3 5000
//' apply(draws, 1:2, mean)        # ~ nu * S
//'
//' @export
// [[Rcpp::export]]
arma::cube rwish(const int n, const double nu, const arma::mat & S) {
  int p = S.n_rows;
  if( nu < p ) Rcpp::stop("Wishart distribution requires nu >= ncols(S).");

  arma::mat Schol = arma::chol(S);
  arma::mat Z_all = arma::randn(nu * n, p) * Schol;   // un solo gemm
  arma::cube out(p, p, n);
  for (int i = 0; i < n; ++i) {
    arma::mat Zi = Z_all.rows(i * nu, (i + 1) * nu - 1);
    out.slice(i) = Zi.t() * Zi;
  }
  return out;
}

//' Sample from a Normal-Inverse-Wishart distribution
//'
//' Draws \code{n} independent samples \eqn{(\mu, \Sigma)} from a
//' Normal-Inverse-Wishart distribution:
//' \deqn{\Sigma \sim \mathcal{IW}(\Psi, \nu), \qquad
//'       \mu \mid \Sigma \sim \mathcal{N}(\xi,\; \Sigma / \kappa).}
//'
//' @param n Integer. Number of independent draws.
//' @param mean Numeric vector of length \eqn{p}: the prior mean \eqn{\xi}.
//' @param Psi Symmetric positive-definite \eqn{p \times p} matrix: scale matrix \eqn{\Psi}.
//' @param kappa Positive scalar: prior precision multiplier on the mean.
//' @param nu Degrees of freedom; must satisfy \code{nu > p - 1}.
//'
//' @return A list with two elements:
//' \describe{
//'   \item{mu}{A \eqn{p \times n} matrix; column \code{i} is the i-th \eqn{\mu} draw.}
//'   \item{Sigma}{A \eqn{p \times p \times n} array; slice \code{[, , i]} is the i-th \eqn{\Sigma} draw.}
//' }
//'
//' @examples
//' p   <- 3
//' sim <- 10000
//' Psi <- diag(p); Psi[1,2] <- Psi[2,1] <- -.5
//' out <- rniw(n = sim, mean = rep(0, p), Psi = Psi, kappa = 1, nu = p + 5)
//'
//' dim(out$mu)                       # p x n
//' dim(out$Sigma)                    # p x p x n
//' rowMeans(out$mu)                  # ~ ex
//' apply(out$Sigma,c(1,2),sum)*4/sim # ~ Psi
//'
//' @export
// [[Rcpp::export]]
Rcpp::List rniw(const int n,
                 const arma::colvec & mean,
                 const arma::mat    & Psi,
                 const double kappa,
                 const double nu) {

   const arma::uword p = Psi.n_rows;

   if (n <= 0)                Rcpp::stop("'n' must be a positive integer.");
   if (Psi.n_cols != p)       Rcpp::stop("'Psi' must be a square matrix.");
   if (mean.n_elem != p)      Rcpp::stop("Length of 'mean' (%u) must match the dimension of 'Psi' (%u).", (unsigned) mean.n_elem, (unsigned) p);
   if (kappa <= 0.0)          Rcpp::stop("'kappa' must be strictly positive.");
   if (nu <= (double)p - 1.0) Rcpp::stop("Degrees of freedom 'nu' (%g) must be greater than p - 1 = %u.", nu, (unsigned)(p - 1));

   const arma::mat PsiInv = arma::inv_sympd(Psi);

   arma::mat  mu_out(p, n);
   arma::cube Sigma_out(p, p, n);

   for (int i = 0; i < n; ++i) {
     arma::mat SigmaInv = cpp_rwish_1(nu, PsiInv);     // Sigma^-1 ~ W(Psi^-1, nu)
     arma::mat Sigma    = arma::inv_sympd(SigmaInv);   // Sigma    ~ IW(Psi, nu)
     mu_out.col(i)      = cpp_mvrnorm_1(mean, Sigma / kappa);
     Sigma_out.slice(i) = Sigma;
   }

   return Rcpp::List::create(
     Rcpp::Named("mu")    = mu_out,
     Rcpp::Named("Sigma") = Sigma_out
   );
 }
