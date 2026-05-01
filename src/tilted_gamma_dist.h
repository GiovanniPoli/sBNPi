#ifndef TILTED_GAMMA_DIST_H
#define TILTED_GAMMA_DIST_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

arma::colvec h_tilde(   const arma::colvec & x,
                        const double A, const double B, const int J);
double h_tilde_1( const double x,
                  const double A,
                  const double B,
                  const int J);
Rcpp::NumericVector h_tilde_prime( const Rcpp::NumericVector & x,
                                   const double A,
                                   const double B,
                                   const int J );
double h_tilde_prime_1( const double x,
                        const double A,
                        const double B,
                        const int J );
double find_mode_tilted_gamma( const double A,
                               const double B,
                               const int J,
                               const double epsilon,
                               const int max_iter );
std::pair<double,double> aux_ar_r_g_tilde_1( const double A,
                                             const double B,
                                             const int J,
                                             const int N);
double cpp_rtilted_gamma_1( const double A,
                            const double B,
                            const int    J,
                            const int    N,
                            const int maxiter);
#endif
