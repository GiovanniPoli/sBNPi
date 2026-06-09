#include <iostream>
#include <chrono>
#include <thread>
#include <RcppArmadillo.h>

#include "utils.h"
#include "hamming_dist.h"
#include "normal_dist.h"
#include "wrapper.h"

//' Pointwise log conditional predictive ordinates from an sBNP MCMC chain
//'
//' Computes the pointwise log conditional predictive ordinate (log-CPO) values
//' as a post-processing step from the MCMC output produced by \code{sBNPi}
//' or \code{sBNPi_alpha}.
//'
//' For each posterior draw, the function evaluates the marginal likelihood
//' contribution of the observed components of each observation, integrating over
//' the truncated mixture components.
//'
//' @param CHAIN A list containing the MCMC output produced by \code{sBNPi} or
//'   \code{sBNPi_alpha}. Each element of the list must contain at least
//'   \code{Pi} and \code{Phi}. If \code{alpha_const = FALSE}, each element must
//'   also contain \code{alpha}.
//' @param data A numeric matrix of dimension \eqn{n \times q} containing the
//'   observed data. Missing values must be encoded as \code{NA}.
//' @param group An integer vector of length \eqn{n} containing the group
//'   membership of each observation. The coding must be consistent with the
//'   indexing used in the MCMC output.
//' @param L Integer. Number of mixture components.
//' @param alpha_const Logical. If \code{TRUE}, the Hamming likelihood parameter
//'   \code{alpha} is treated as fixed. If \code{FALSE}, \code{alpha} is read
//'   from each posterior sample in \code{CHAIN}.
//' @param alpha Numeric. Fixed value of the Hamming likelihood parameter, used
//'   only when \code{alpha_const = TRUE}. It must belong to \eqn{(0, 0.5)}.
//' @return A numeric vector of length \eqn{n} containing the log-CPO values.
//'
//' @export
// [[Rcpp::export]]
arma::vec logCPO( const Rcpp::List & CHAIN, const arma::mat  & data,
                 const arma::ivec & group, const int L,
                 bool  alpha_const = false, double alpha = -1.0 ){

  int S  = CHAIN.size();
  int NN = data.n_rows;
  int q  = data.n_cols;

  if(alpha_const && ((alpha <= 0.0) || (alpha >= 0.5))){
    Rcpp::stop("'alpha_const' = TRUE requires alpha in (0, 0.5).");
  }

  arma::vec inv_cpo_sum(NN, arma::fill::zeros);

  arma::uvec   idx_obs;
  arma::ivec   r_i;
  arma::colvec yi;
  arma::colvec yobs_i;

  Rcpp::List sample;
  arma::mat  Pi_s;
  Rcpp::List Phi_s;

  arma::mat    Sigma_l;
  arma::colvec mu_l;
  arma::ivec   c_l;

  arma::mat    Sigma_obs_l;
  arma::colvec mu_obs_l;

  arma::vec    log_comp(L);

  double lhm_sl;
  double lqN_sl;
  double loglik_is;

  int j;
  double m ;

  Rcpp::Rcout << "Evaluating log(CPO) for each subject, post processing the MCMC chain..." << std::endl ;
  for(int s = 0; s < S; ++s){
    Rcpp::Rcout << "\r[" << s << "/" << S << "]                        ";

    sample = Rcpp::as<Rcpp::List>(CHAIN[s]);
    Pi_s   = Rcpp::as<arma::mat>(sample["Pi"]);
    Phi_s  = Rcpp::as<Rcpp::List>(sample["Phi"]);

    if(!alpha_const){
      alpha = Rcpp::as<double>(sample["alpha"]);
    }

    for(int i = 0; i < NN; ++i){
      j = group(i);

      yi      = data.row(i).as_col();
      idx_obs = arma::find_finite(yi);
      yobs_i  = yi.elem(idx_obs);

      r_i = arma::ivec(q, arma::fill::zeros);
      r_i.elem(idx_obs).ones();

      for(int l = 0; l < L; ++l){
        Rcpp::List Phi_l = Rcpp::as<Rcpp::List>(Phi_s[l]);
        Sigma_l = Rcpp::as<arma::mat>(Phi_l["Sigma"]);
        mu_l    = Rcpp::as<arma::vec>(Phi_l["mu"]);
        c_l     = Rcpp::as<arma::ivec>(Phi_l["c"]);

        Sigma_obs_l = Sigma_l.submat(idx_obs, idx_obs);
        mu_obs_l    = mu_l.elem(idx_obs);

        lhm_sl = log_hamming_1_obs(r_i, c_l, alpha);
        lqN_sl = log_dmvn_mat_1_obs(yobs_i, mu_obs_l, Sigma_obs_l);

        log_comp(l) = std::log(Pi_s(l, j)) + lhm_sl + lqN_sl;
      }

      m               = log_comp.max();
      loglik_is       = m + std::log(arma::sum(arma::exp(log_comp - m)));;

      inv_cpo_sum(i) += std::exp(-loglik_is);
    }
  }
  arma::vec log_cpo = -arma::log(inv_cpo_sum / S);
  Rcpp::Rcout << "\r[" << S << "/" << S << "]                        "  << std::endl;
  return log_cpo;
}

//' Log pseudo marginal likelihood from an sBNP MCMC chain
//'
//' Computes the log pseudo marginal likelihood (LPML) as a post-processing step
//' from the MCMC output produced by \code{sBNPi} or \code{sBNPi_alpha}.
//'
//' The LPML is obtained by summing the log conditional predictive
//' ordinate values returned by \code{logML}.
//'
//' @param CHAIN A list containing the MCMC output produced by \code{sBNPi} or
//'   \code{sBNPi_alpha}. Each element of the list must contain at least
//'   \code{Pi} and \code{Phi}. If \code{alpha_const = FALSE}, each element must
//'   also contain \code{alpha}.
//' @param data A numeric matrix of dimension \eqn{n \times q} containing the
//'   observed data. Missing values must be encoded as \code{NA}.
//' @param group An integer vector of length \eqn{n} containing the group
//'   membership of each observation. The coding must be consistent with the
//'   indexing used in the MCMC output.
//' @param L Integer. Number of mixture components.
//' @param alpha_const Logical. If \code{TRUE}, the Hamming likelihood parameter
//'   \code{alpha} is treated as fixed. If \code{FALSE}, \code{alpha} is read
//'   from each posterior sample in \code{CHAIN}.
//' @param alpha Numeric. Fixed value of the Hamming likelihood parameter, used
//'   only when \code{alpha_const = TRUE}. It must belong to \eqn{(0, 0.5)}.
//'
//' @return A numeric scalar containing the estimated log pseudo marginal
//'   likelihood.
//'
//' @seealso \code{\link{sBNPi}}, \code{\link{sBNPi_alpha}}, \code{\link{logML}}
//'
//' @export
// [[Rcpp::export]]
double logPML( const Rcpp::List & CHAIN, const arma::mat  & data,
                  const arma::ivec & group, const int L,
                  bool  alpha_const = false, double alpha = -1.0 ){

  arma::vec log_cpo = logCPO( CHAIN, data, group, L, alpha_const, alpha );
  return  arma::sum(log_cpo);
}

