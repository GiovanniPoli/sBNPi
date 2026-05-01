#ifndef HAMMING_DIST_H
#define HAMMING_DIST_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>


double log_hamming_1_obs( const arma::ivec& r_i,
                          const arma::ivec& c,
                          double alpha );
arma::ivec cpp_rmvbern(const arma::vec & p);
arma::vec post_param_hamming(
    const arma::ivec &c0,
    double alpha0,
    double alpha,
    const  arma::mat & r_mat);

#endif
