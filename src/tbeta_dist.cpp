#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "tbeta_dist.h"

double cpp_rtbeta_1(const double a, const double b) {
  double Fd = p_4beta(0.5, a, b, 0.0, 1.0);
  double u  = arma::randu();
  return q_4beta(u * Fd, a, b, 0.0, 1.0);
}
