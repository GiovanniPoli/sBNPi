#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "dirichlet_dist.h"

arma::colvec cpp_rdirichlet_1(const arma::colvec & parameters){
  int p = parameters.n_elem ;
  arma::colvec ret(p);
  for(int i = 0; i<p; i++){
    ret(i) = arma::randg<double>(arma::distr_param(parameters(i),1.0));
  }
  ret = ret/arma::sum(ret);
  return ret;
}
