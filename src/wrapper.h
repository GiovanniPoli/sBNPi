#ifndef WRAPPER_H
#define WRAPPER_H

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "param_struct.h"

Rcpp::List wrap_param_cluster(const std::vector<param_cluster>& PHI);

#endif
