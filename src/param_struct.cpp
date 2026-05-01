#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "param_struct.h"

param_cluster::param_cluster( ){
  c        = arma::ivec();
  mu       = arma::colvec();
  Sigma    = arma::mat();
  SigmaInv = arma::mat();
}


param_cluster::param_cluster( const arma::ivec   & c_init,
                              const arma::colvec & mu_init,
                              const arma::mat    & Sigma_init){
  c        = c_init;
  mu       = mu_init;
  Sigma    = Sigma_init;
  SigmaInv = arma::inv_sympd(Sigma_init);
};


param_cluster::param_cluster( const arma::ivec   & c_init,
                              const arma::colvec & mu_init,
                              const arma::mat    & Sigma_init,
                              const arma::mat    & SigmaInv_init ){
  c        = c_init;
  mu       = mu_init;
  Sigma    = Sigma_init;
  SigmaInv = SigmaInv_init;
};

