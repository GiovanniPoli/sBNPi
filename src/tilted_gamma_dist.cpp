#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>

#include "tilted_gamma_dist.h"

// Tilted Gamma as described in:
// Blocked Gibbs sampler for hierarchical Dirichlet processes.
// Das, S. et al.(2025),

arma::colvec h_tilde(   const arma::colvec & x,
                        const double A, const double B, const int J){
  return - double(J) * arma::lgamma(x) + (A-1.0) *arma::log(x) - B*x ;
}

double h_tilde_1( const double x,
                  const double A,
                  const double B,
                  const int J){
  return (-double(J)*std::lgamma(x)) - (B*x) + ((A-1.0)*std::log(x)) ;
}

Rcpp::NumericVector h_tilde_prime(
    const Rcpp::NumericVector & x,
    const double A,
    const double B,
    const int J ){
  return   - double(J) *  Rcpp::digamma(x) + (A-1.0)/x- B;
}
double h_tilde_prime_1(
    const double x,
    const double A,
    const double B,
    const int J ){
  Rcpp::NumericVector xv = Rcpp::NumericVector::create(x);
  double digamma_val = Rcpp::digamma(xv)[0] ;
  return -double(J) *digamma_val + (A-1.0)/x- B;
}
double find_mode_tilted_gamma(
    const double A,
    const double B,
    const int J,
    const double epsilon,
    const int max_iter )  {
  // Bisection method for tilted gamma.
  // return root that implies the tilted gamma mode
  double M = (B > 0.0) ? 1.5 : std::exp(1.0 - B/double(J));
  double a  = epsilon ;
  double b  = M-epsilon ;

  double fa = h_tilde_prime_1(   a,  A, B, J);
  double fb = h_tilde_prime_1(   b,  A, B, J);

  if (fa * fb >= 0.0) {
    double ret = (fa >= fb) ? b : a  ;
    return ret;
  }
  double c, fc;
  int iter = 0;

  while ( ((b - a)/2.0 > epsilon) && (iter < max_iter)) {
    c  = (a + b) / 2.0;
    fc = h_tilde_prime_1(c, A, B, J);
    if (fa * fc < 0.0) {
      b  = c;
      fb = fc;
    } else {
      a  = c;
      fa = fc;
    }
    iter++;
  }
  return (a + b) / 2.0;
}


std::pair<double,double> aux_ar_r_g_tilde_1(
    const double A, const double B,
    const int J,    const int N){
  // AR-Algorithm from Das, S. et al.(2025),
  // Blocked Gibbs sampler for hierarchical Dirichlet processes.
  arma::colvec u = arma::randu(2);
  int n_knots = 2*N+2;
  Rcpp::NumericVector knots(n_knots);
  double    m = find_mode_tilted_gamma(A,B,J, 1e-6, 100 );
  double last = (B > 0.0) ? m + 1.5 : std::exp(1.0 - B / double(J));

  knots[N]     = m;
  knots[0]     = m/2.0;
  knots[2*N]   = (m + last) /2.0;
  knots[2*N+1] = last;

  double step  = m/2.0/N ;
  for(int i = 1; i<N; ++i){
    knots[i] =  m/2.0 + double(i)*m/2.0/N;  // i = 1, ... , (N-2)
  }
  step = ((m + last)/2.0 - m)/ N ;
  for(int i = 1; i<N; ++i){
    knots[N+i] = m + double(i) * step  ;
  }

  arma::colvec      a   = h_tilde(knots, A, B, J);
  arma::colvec lambda   = h_tilde_prime(knots, A, B, J);

  arma::colvec       am = knots;
  arma::colvec q(2*N+3);
  q(0)     = 0.0;
  q(2*N+2) = arma::datum::inf;

  q.subvec(1, 2*N+1) = ( a.subvec(1, 2*N+1)  -
    a.subvec(0,  2*N)   +
    am.subvec(0, 2*N)   % lambda.subvec(0, 2*N) -
    am.subvec(1, 2*N+1) % lambda.subvec(1, 2*N+1) ) / ( lambda.subvec(0,2*N) - lambda.subvec(1, 2*N+1) );

  arma::colvec C = 1.0 / lambda % arma::exp( a  - am%lambda ) % ( arma::exp(q.subvec(1,2*N+2)%lambda) -
                   arma::exp(q.subvec(0,2*N+1)%lambda) ) ;

  C =  arma::cumsum( C / arma::sum(C) );
  double x;
  double loggtildex;
  for(int i = 0; i< C.n_elem; ++i ){
    if( u(0) < C[i]){
      if(i == N){
        x       =  q(i)+ (q(i+1)-q(i))*u(1) ;
      }else{
        x =  1/lambda(i)* std::log( u(1) * std::exp(lambda(i) * q(i+1)) + (1-u(1)) * std::exp( lambda(i)*q(i)) ) ;
      }
      loggtildex = a(i) + lambda(i) * ( x - am(i) );
      return std::make_pair( x, loggtildex );
    }
  }
}


double cpp_rtilted_gamma_1( const double A,
                            const double B,
                            const int    J,
                            const int    N,
                            const int maxiter ){
  std::pair<double,double> pair ;
  double sample = - 1.0;
  double logfx ;

  for(int i =0; i<maxiter; ++i){
    double logu = std::log(arma::randu<double>());
    pair        = aux_ar_r_g_tilde_1(A, B, J, N);
    logfx       = h_tilde_1(pair.first, A,B, J );

    if( logu < (logfx - pair.second) ){
      return pair.first ;
    }
  }
  Rcpp::warning("AR procedure exits with maxiter ");
  return 1.0 ;
}
