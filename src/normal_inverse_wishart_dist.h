#ifndef NORMAL_INVERSE_WISHART_H
#define NORMAL_INVERSE_WISHART_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>


arma::mat cpp_rwish_1( const double nu, const arma::mat & S);
std::tuple<arma::colvec, arma::mat, arma::mat> rniw_cpp(const arma::colvec & ex,
                                                        const arma::mat    & PsiInv,
                                                        const double kappa,
                                                        const double nu);
std::tuple<arma::colvec, double, double, arma::mat> post_param_niw( const  arma::mat    & X,
                                                                    const  arma::colvec & m0,
                                                                    const  arma::mat& Psi0,
                                                                    double kappa0,
                                                                    double nu0 );
#endif
