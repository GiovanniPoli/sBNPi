#ifndef PARAM_STRUCT_H
#define PARAM_STRUCT_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

struct param_cluster{
  arma::ivec    c ;
  arma::colvec  mu  ;
  arma::mat     Sigma ;
  arma::mat     SigmaInv ;

  // Empty
  param_cluster( );
  // Manual
  param_cluster( const arma::ivec   & c_init,
                 const arma::colvec & mu_init,
                 const arma::mat    & Sigma_init ) ;
  // Manual (Complete)
  param_cluster( const arma::ivec   & c_init,
                 const arma::colvec & mu_init,
                 const arma::mat    & Sigma_init,
                 const arma::mat    & SigmaInv_init) ;
};



#endif
