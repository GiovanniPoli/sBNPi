#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "normal_dist.h"
#include "normal_inverse_wishart_dist.h"
#include "hamming_dist.h"
#include "param_struct.h"
#include "tilted_gamma_dist.h"

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// update_imputation_v2
void missing_data_augmentation( arma::subview_row<double>  yi,
                                const arma::uvec & missing_idx,
                                const arma::uvec & obs_idx,
                                const arma::colvec & mu,
                                const arma::mat & Sigma ){
  arma::colvec yi_new = yi.t();

  arma::colvec yobs   = yi_new(obs_idx);
  const arma::mat S11 = Sigma.submat(missing_idx, missing_idx);
  const arma::mat S12 = Sigma.submat(missing_idx, obs_idx);
  const arma::colvec mu1 = mu(missing_idx);
  const arma::colvec mu2 = mu(obs_idx);

  arma::mat S22 = Sigma.submat(obs_idx, obs_idx);

  const arma::mat    B          = S12*arma::inv_sympd(S22);
  const arma::colvec mean_cond  = mu1 + B * (yobs - mu2);
  arma::mat Sigma_cond          = S11 - B * S12.t();

  // for computational stability
  Sigma_cond.diag() += 1e-10 * arma::trace(Sigma_cond) / Sigma_cond.n_rows;
  Sigma_cond         = arma::symmatu(Sigma_cond);

  // imputation
  yi_new(missing_idx) = cpp_mvrnorm_1( mean_cond, Sigma_cond );
  yi = yi_new.t();
}

// update_Z_ji_cpp()
int update_cluster_membership(const arma::vec  & Pi_j,
                              const arma::vec  & y_i,
                              const arma::ivec & r_i,
                              const std::vector<param_cluster> & phi,
                              const double & alpha) {

  const arma::uword L = phi.size();

  arma::ivec       cluster_labels = arma::regspace<arma::ivec>(0, L - 1);

  arma::vec F_ji(L);

  for (arma::uword l = 0; l < L; ++l) {
    const param_cluster& phi_l = phi[l];
    F_ji[l] = log_dmvn_both_mat_1_obs(y_i, phi_l.mu, phi_l.SigmaInv, phi_l.Sigma) +
              log_hamming_1_obs(r_i, phi_l.c, alpha);
  }

  const arma::vec log_prob = arma::log(Pi_j) + F_ji;
  const double m = log_prob.max();
  arma::vec unnormalized_weights = arma::exp(log_prob - m);

  int  new_label = Rcpp::RcppArmadillo::sample(cluster_labels, 1, false, unnormalized_weights)(0);
  return new_label;
}

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
                const double & kappa0 ){

  if(Yc_submat.n_rows == 0){
    std::tuple<arma::colvec, arma::mat, arma::mat> niw_draw = rniw_cpp(m0, Sigma0Inv, kappa0, nu0);

    arma::vec p_c = arma::conv_to<arma::vec>::from(c0 == 0);
    p_c           = p_c * alpha0 + (1.0 - p_c) * (1.0 - alpha0);
    c_it = cpp_rmvbern(p_c);

    mu_it       = std::get<0>(niw_draw);
    SigmaInv_it = std::get<1>(niw_draw);
    Sigma_it    = std::get<2>(niw_draw);

  }else{
    arma::vec new_c_param = post_param_hamming( c0, alpha0, alpha, RR_submat );
    c_it = cpp_rmvbern(new_c_param);

    std::tuple<arma::vec, double, double, arma::mat> niw_post = post_param_niw(Yc_submat, m0, Sigma0, kappa0, nu0);

    arma::vec m_n        = std::get<0>(niw_post);
    double kappa_n       = std::get<1>(niw_post);
    double nu_n          = std::get<2>(niw_post);
    arma::mat Sigma_n    = std::get<3>(niw_post);
    arma::mat SigmaInv_n = arma::inv_sympd(Sigma_n);

    std::tuple<arma::vec, arma::mat, arma::mat> niw_draw = rniw_cpp(m_n, SigmaInv_n, kappa_n, nu_n);

    mu_it       = std::get<0>(niw_draw);
    SigmaInv_it = std::get<1>(niw_draw);
    Sigma_it    = std::get<2>(niw_draw);
  }
}
