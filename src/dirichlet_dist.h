#ifndef DIRICHLET_DIST_H
#define DIRICHLET_DIST_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

arma::colvec cpp_rdirichlet_1(const arma::colvec & parameters);

#endif
