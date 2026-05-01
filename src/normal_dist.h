#ifndef NORMAL_DIST_H
#define NORMAL_DIST_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

arma::colvec cpp_mvrnorm_1(const arma::colvec & mu, const arma::mat & sigma);
double log_dmvn_both_mat_1_obs( const arma::colvec & x, const arma::colvec & mu, const arma::mat    & SigmaInv, const arma::mat    & Sigma);

#endif
