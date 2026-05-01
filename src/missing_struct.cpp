#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "missing_struct.h"

missing::missing() {
  obs = arma::uvec() ;
  mis = arma::uvec() ;
  n_mis  = 0 ;
  subject_idx = 0 ;
}

missing::missing( const arma::colvec & y, const arma::uword idx ) {
  mis = arma::find_nonfinite(y);
  obs = arma::find_finite(y);
  n_mis = mis.n_elem ;
  subject_idx = idx ;
}
