#ifndef MISSING_STRUCT_H
#define MISSING_STRUCT_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

struct missing {
  arma::uvec   mis ;
  arma::uvec   obs ;
  arma::uword  n_mis  ;
  arma::uword  subject_idx ;

  missing();
  missing( const arma::colvec & y, const arma::uword idx );
};

#endif
