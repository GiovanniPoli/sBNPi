#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "param_struct.h"

Rcpp::List wrap_param_cluster(const std::vector<param_cluster>& PHI) {
  int L = PHI.size();
  Rcpp::List out(L);

  for (int l = 0; l < L; ++l) {

    const param_cluster &  par_it      = PHI[l];
    arma::ivec c_it        = par_it.c ;
    arma::vec  mu_it       = par_it.mu ;
    arma::mat  Sigma_it    = par_it.Sigma;

    out[l] = Rcpp::List::create(
      Rcpp::Named("mu")    = mu_it,
      Rcpp::Named("Sigma") = Sigma_it,
      Rcpp::Named("c")     = c_it
    );
  }
  return out;
}
