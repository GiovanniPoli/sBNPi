#ifndef UPDATE_STEPS_H
#define UPDATE_STEPS_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

void missing_data_augmentation( arma::subview_row<double>  yi,
                                const arma::uvec & missing_idx,
                                const arma::uvec & obs_idx,
                                const arma::colvec & mu,
                                const arma::mat & Sigma );

void update_phi(const arma::mat& Yc_submat,
                const arma::mat& RR_submat,
                arma::ivec &  c_it,
                arma::vec  &  mu_it,
                arma::mat  &  Sigma_it,
                arma::mat  &  SigmaInv_it,
                const double & alpha,
                const arma::ivec & c0,
                const double & alpha0,
                const arma::mat & Sigma0Inv,
                const arma::mat & Sigma0,
                const arma::vec & m0,
                const double & nu0,
                const double & kappa0 );

int update_cluster_membership(const arma::vec  & Pi_j,
                              const arma::vec  & y_i,
                              const arma::ivec & r_i,
                              const std::vector<param_cluster> & phi,
                              const double & alpha);

#endif
