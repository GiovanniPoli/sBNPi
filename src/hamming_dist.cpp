#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "hamming_dist.h"

double log_hamming_1_obs( const arma::ivec& r_i,
                          const arma::ivec& c,
                          double alpha ){
  int q = r_i.n_elem;
  int hamming_dist = arma::accu(r_i != c);
  return hamming_dist * std::log(alpha) + (q - hamming_dist) * std::log(1.0 - alpha);
}

arma::ivec cpp_rmvbern(const arma::vec & p){
  arma::vec u = arma::randu<arma::vec>(p.n_elem);      // uniform [0,1)
  return arma::conv_to<arma::ivec>::from(u < p);       // elementwise comparison
}

arma::vec post_param_hamming(
    const arma::ivec &c0,
    double alpha0,
    double alpha,
    const  arma::mat & r_mat) {

  int n_k = r_mat.n_rows;
  int q   = r_mat.n_cols;
  arma::rowvec r_sum = arma::sum(r_mat, 0); // column sums

  arma::vec p_vec(q);
  for (int l = 0; l < q; l++) {
    double term1 = std::pow(alpha0/(1.0-alpha0), 2.0*c0[l] - 1.0);
    double term2 = std::pow(alpha/(1.0-alpha), 2.0*(r_sum[l] - n_k/2.0));
    p_vec[l] = 1.0 / (1.0 + term1 * term2);
  }
  return p_vec ;
}

//' Sample independent multivariate Bernoulli vectors
//'
//' Draws \code{n} independent samples; each sample is a length-\eqn{q}
//' vector whose \eqn{j}-th component is independently Bernoulli with
//' probability \code{p[j]}.
//'
//' @param n Integer. Number of draws.
//' @param p Numeric vector of length \eqn{q}; each entry must lie in
//'   \eqn{[0, 1]}.
//'
//' @return An integer matrix of dimension \eqn{n \times q}; row \code{s}
//'   is the s-th draw, and entry \code{[s, j]} equals 1 with probability
//'   \code{p[j]} and 0 otherwise.
//'
//' @examples
//' p <- c(0.1, 0.5, 0.9)
//' X <- rmvbern(n = 10000, p = p)
//' colMeans(X)   # ~ p
//'
//' @export
// [[Rcpp::export]]
arma::imat rmvbern(const int n, const arma::vec & p) {
  if (p.min() < 0.0 || p.max() > 1.0) Rcpp::stop("All entries of 'p' must be in [0, 1].");
  arma::mat U = arma::randu(n, p.n_elem);
  U.each_row() -= p.t();
  return arma::conv_to<arma::imat>::from(U < 0.0);
}
