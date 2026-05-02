#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <tuple>

#include "utils.h"
#include "missing_struct.h"
#include "param_struct.h"
#include "update_steps.h"
#include "tbeta_dist.h"
#include "dirichlet_dist.h"
#include "tilted_gamma_dist.h"
#include "wrapper.h"

//' Hierarchical BNP imputation via MCMC
//'
//' Runs an MCMC algorithm for a structural Bayesian nonparametric missing-imputation (sBNPi).
//'
//' @param data Numeric matrix (\eqn{n} x \eqn{q}) of observed data. Missing values must be
//'   encoded as not-available (i.e., \code{NA}). The dataset is a assumed the result of a
//'   data integration of \eqn{J} different data sources each with \eqn{n_j} observations.
//'   Total sample size is thus \eqn{n=\sum_{j=1}^J n_j}.
//' @param group Integer vector of length \eqn{n} indicating group membership for
//'   each observation. Input is required to assume values in \eqn{\{0, \ldots, J-1\}}.
//' @param a0,b0 Positive scalars. Hyperparameters for the truncate Beta prior on \eqn{\alpha}.
//' @param c0,alpha0 Hyperparameters of the base-measure component for the reference pattern \eqn{c\sim}Hamming\eqn{(\alpha_0,c_0)}.
//'   \code{c0} is a binary vector of length \eqn{q} representing the prior reference pattern.
//'   \code{alpha0} is a dispersion parameter in \eqn{(0,\frac{1}{2})} around the centroid \code{c0}.
//' @param mu0,kappa0,Sigma0,nu0 Hyperparameters of the base-measure component for the complete data parameters \eqn{\mu} and \eqn{\Sigma}, with \eqn{\mu\mid\Sigma\sim \mathcal{N}}\eqn{(\mu_0,\kappa_0^{-1}\Sigma)} and
//'   \eqn{\Sigma\sim \mathcal{W}^{-1}}\eqn{(\nu_0,\Sigma_0)}. \code{mu0} is a vector, \code{Sigma0} is a matrix \eqn{q} x \eqn{q} matrix,
//'   while \code{kappa0} and \code{nu0} are positive scalars.
//'   Degrees of freedom \eqn{\nu_0} must be greater than \eqn{q-1}.
//' @param gamma0,eta0 Positive scalars. Hyperparameter for gamma prior on the DP precision parameter.
//' @param L Integer. Number of mixture components (finite truncation level).
//' @param sample Integer. Number of MCMC samples to store.
//' @param burn Integer. Number of burn-in iterations.
//' @param thinning Integer. Thinning interval.
//' @param cluster_init Initialization strategy for the cluster allocations. One
//'   of \code{"random"}, \code{"cluster"}, \code{"singleton"}, or \code{"manual"}.
//' @param allocation_init \code{NULL} or an integer vector used when
//'   \code{cluster_init = "manual"}.
//' @param approx_tilted_gamma Integer. Number of splits controlling approximation for
//'   tilted-gamma sampling (See Das et al., 2025).
//' @param maxiter_tilted_gamma Integer. Maximum iterations for tilted-gamma mode solver (See Das et al., 2025).
//'
//' @return A list of length \code{sample}. \code{List}'s element contains:
//' \describe{
//'   \item{\code{Imputation}}{Input \eqn{n} x \eqn{q} data matrix with \code{NA} filled with imputed values.}
//'   \item{\code{Z}}{A cluster allocation vector of size \eqn{n}.}
//'   \item{\code{Phi}}{A list of cluster-specific parameters. Each element is a \code{List} of three elements \code{Phi$mu}, \code{Phi$Sigma} and \code{Phi$c}}
//'   \item{\code{Pi}}{An \eqn{L} x \eqn{J} matrix. Each column represents group-specific mixture weights.}
//'   \item{\code{Beta}}{A vector of \code{length} \eqn{L}. It represents the global-measure \eqn{G_0}'s mixture weights.}
//'   \item{\code{alpha}}{Scalar parameter controlling missingness structure.}
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//'
//' # Simulate a small dataset from J = 2 groups, q = 3 variables
//' n <- 120
//' q <- 3
//' J <- 2
//' group <- sample(0:(J - 1), n, replace = TRUE)
//'
//' # Two well-separated mixture components
//' mu1 <- c(-2, -2,  0)
//' mu2 <- c( 2,  2,  0)
//' z   <- sample(1:2, n, replace = TRUE)
//' Y   <- t(sapply(z, function(k) {
//'   if (k == 1) rnorm(q, mean = mu1) else rnorm(q, mean = mu2)
//' }))
//'
//' # Introduce ~20% missingness completely at random
//' miss_idx <- sample(length(Y), size = round(0.20 * length(Y)))
//' Y[miss_idx] <- NA
//'
//' # Hyperparameters
//' c0     <- rep(0L, q)
//' alpha0 <- 0.1            # must lie in (0, 1/2)
//' mu0    <- rep(0, q)
//' Sigma0 <- diag(q)
//' kappa0 <- 1
//' nu0    <- q + 2          # must be > q - 1
//'
//' # Run a short MCMC (small sample/burn for illustration only)
//' post <- sBNPi(data     = Y,
//'               group    = group,
//'               a0       = 1,    b0    = 1,
//'               c0       = c0,   alpha0 = alpha0,
//'               Sigma0   = Sigma0, mu0  = mu0,
//'               kappa0   = kappa0, nu0  = nu0,
//'               gamma0   = 1,    eta0  = 1,
//'               L        = 10,
//'               sample   = 1000,  burn  = 1000, thinning = 10)
//'
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List sBNPi( const arma::mat  & data,
                  const arma::ivec & group, // index j
                  const double a0,        const double b0, // prior for alpha
                  const arma::ivec  & c0, const double & alpha0,
                  const arma::mat  & Sigma0,   const arma::vec & mu0,
                  const double kappa0,         const double nu0,
                  const double gamma0, const double eta0, // b0 in Snigdha Das, Yabo Niu, Yang Ni, Bani K. Mallick, Debdeep Pati (2025).
                  const int L = 10,
                  const unsigned int sample = 1000, const unsigned int burn = 0, const unsigned int thinning = 1,
                  const std::string cluster_init = "random",
                  const Rcpp::Nullable<arma::ivec> allocation_init = R_NilValue,
                  const int approx_tilted_gamma  = 1, const int maxiter_tilted_gamma = 1e3){

  // Index for cycles, useful indeces,
  arma::ivec r_i;
  arma::vec  y_i;
  arma::vec  Pi_j;

  arma::ivec unique =  arma::unique(group);
  const int J  = unique.n_elem ;
  const int q  = data.n_cols  ;
  const int NN = data.n_rows ;
  const int mL = L-1 ;

  int i, ii, j, l;
  // YC:  Augmented data
  // R:   Missing indicator
  // Dim: J x mL
  arma::mat RR( NN, q, arma::fill::ones) ;
  arma::mat  augY = data ;
  arma::uvec idx_mis;
  arma::uvec idx_obs;
  for( int var = 0 ; var < q; ++var){
    // iter
    arma::colvec     ycol = augY.col(var);
    arma::colvec     rcol = RR.col(var);
    // idx
    arma::uvec idx_mis = arma::find_nonfinite(ycol);
    arma::uvec idx_obs = arma::find_finite(ycol);
    if(!idx_mis.is_empty() && !idx_obs.is_empty()){
      double col_mean = arma::mean(ycol(idx_obs));
      ycol(idx_mis).fill(col_mean);
      rcol(idx_mis).zeros();
    }
    augY.col(var) = ycol;
    RR.col(var)   = rcol;
  }
  // Iterator_for_missing: std::vector to cycle with infnormations for data augmentation step
  std::vector<missing> iterator_for_missing ;
  missing mis_map_it ;
  for( i = 0 ; i < NN; ++ i){
    mis_map_it = missing(data.row(i).as_col(), i) ;
    if( mis_map_it.n_mis != 0 ){
      iterator_for_missing.push_back( mis_map_it ) ;
    }
  }

  // Z: Allocatios
  //    arma::vec of length NN with values from 1 to L the position is the same of `group` (j)
  arma::ivec Z;

  if (cluster_init == "random") {
    Z = arma::randi<arma::ivec>(NN, arma::distr_param(0,mL));
  } else if (cluster_init == "cluster") {
    // TO DO
    Rcpp::stop("cluster implementation is to do");
  } else if (cluster_init == "singleton") {
    if( NN <= L ){
      Z = arma::regspace<arma::ivec>(0, NN - 1);
    }else{
      Rcpp::stop("Error: cluster_init = 'singleton' requires L greater than or equal to the sample size.");
    }
  } else if (cluster_init == "manual") {
    if (allocation_init.isNull()) {
      Rcpp::stop("Error: init = 'manual' requires non NULL allocation_init, you must provide it.");
    }
    Z = Rcpp::as<arma::ivec>(allocation_init);
    if ((int) Z.n_elem != NN) {
      Rcpp::stop("Error: init_rho must have length equal to nrow(data).");
    }
    // TO DO: add controll for values in \{0, ..., L\}
  } else {
    Rcpp::stop("Invalid init. Use one of: 'random', 'cluster', 'singleton', 'manual'.");
  }

  // PHI: Parameters for finite approximation
  //      length = J*L
  std::vector<param_cluster> PHI(L);

  arma::mat Sigma_init;
  arma::colvec mu_init;
  arma::ivec   c_init;
  Sigma_init.eye(q,q);

  for( int l = 0; l < L ; ++l){
    param_cluster & par = PHI[l];
    c_init  = arma::randi(q, arma::distr_param(0,1));
    mu_init = arma::randn(q) ;
    par = param_cluster( c_init, mu_init,  Sigma_init);
  }

  // create Pi (vector of group specific probabilities)
  arma::mat Pi(L,J);
  Pi.fill(1.0/L);

  // create Sufficient Statistics to update Pi
  arma::mat Pi_ss(L,J, arma::fill::zeros) ;
  for( int ii = 0; ii < NN ;  ++ii ){
    l = Z(ii) ;
    j = group(ii) ;
    Pi_ss(l,j) +=1 ;
  }
  arma::mat    Pi_par(L,J) ;
  arma::colvec Dir_par(L) ;

  // arma::colvec T, length = L
  arma::colvec    T(L, arma::fill::ones);
  double sumT = arma::accu(T);
  arma::colvec Beta = T / sumT ;

  // arma::colvec U (arma vector, length = J
  arma::colvec U(J, arma::fill::ones);
  double ulogsum = 0.0 ;

  arma::colvec B = eta0 - arma::sum(arma::log(Pi), 1) - ulogsum ;

  const double A = gamma0/L;
  double Bk, tj;

  // double alpha
  double alpha = 1.0/4.0;
  double a_prime;
  double b_prime;
  double sumHD ;
  arma::mat Ck_rep ;

  // MCMC Algorithm
  Rcpp::List return_list(sample);
  arma::rowvec Pisum  ; // somme per colonna
  arma::mat    Pi_out ;
  const unsigned int S = burn + thinning * sample;
  unsigned ss = 0;

  const arma::mat Sigma0Inv = arma::inv_sympd(Sigma0);

  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC iterations. (Done)" << std::endl ;
  MyTimePoint t0 = myClock::now();

  for( unsigned int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;

    // 1: Missing Imputation
    for ( auto it  = iterator_for_missing.begin();
          it != iterator_for_missing.end();
          ++it) {
      // mapping with index
      ii  = (it->subject_idx) ;
      j   = group(ii);
      l   = Z(ii)    ;
      // c++ references
      arma::subview_row<double> yi_it = augY.row(ii);
      param_cluster & par_it = PHI[l];
      // Call update
      missing_data_augmentation( yi_it, (it->mis), (it->obs), par_it.mu, par_it.Sigma ) ;
    }

    // 2: Update Z
    for(ii=0; ii < NN; ++ii){
      // mapping with idex
      j    = group(ii);
      r_i  = arma::conv_to<arma::ivec>::from( RR.row(ii).t() );;
      y_i  = augY.row(ii).t();
      Pi_j = Pi.col(j);
      int & z_ji = Z(ii) ;
      // Call update
      Pi_ss( z_ji, j) -= 1 ;
      z_ji = update_cluster_membership(Pi_j, y_i, r_i, PHI, alpha);
      Pi_ss( z_ji, j) += 1 ;
    }

    // 3: Update Phi
    sumHD = 0;
    for(l=0; l < L; ++l){
      // References
      param_cluster &  par_it      = PHI[l];
      arma::ivec    &  c_it        = par_it.c ;
      arma::vec     &  mu_it       = par_it.mu ;
      arma::mat     &  Sigma_it    = par_it.Sigma;
      arma::mat     &  SigmaInv_it = par_it.SigmaInv;

      arma::uvec    idx_l = arma::find(Z==l);
      arma::mat Yc_submat = augY.rows(idx_l);
      arma::mat RR_submat = RR.rows(idx_l);

      update_phi( Yc_submat, RR_submat,
                   c_it, mu_it, Sigma_it, SigmaInv_it,
                   alpha, c0, alpha0, Sigma0Inv, Sigma0, mu0, nu0, kappa0);

      Ck_rep  = arma::conv_to<arma::mat>::from(arma::repmat(c_it.t(), RR_submat.n_rows, 1 ));
      sumHD  += arma::accu(arma::abs(Ck_rep-RR_submat)) ;
    }

    // 4: Update alpha
    a_prime = a0 + sumHD;
    b_prime = b0 + NN*q - sumHD;
    alpha   = cpp_rtbeta_1(a_prime, b_prime);

    // 4: For (j in 1,...,J) update riga J di Matrice Pi (dirichlet)
    for(j=0; j<J ; ++j){
      Dir_par   = Pi_ss.col(j) + T ;
      Pi.col(j) = cpp_rdirichlet_1(Dir_par);
    }
    Pi.clamp(1e-6, 1-1e-6) ;
    ulogsum = arma::accu(arma::log(U)) ;
    B       = eta0 - arma::sum( arma::log(Pi), 1) - ulogsum ;

    // 5: For (k in 1,...,L) update elemento t_j del vettore T
    for( l = 0 ; l < L ; ++l ){
      Bk = std::min( B(l), 1e8 );
      tj = cpp_rtilted_gamma_1(  A, Bk, J, approx_tilted_gamma,  maxiter_tilted_gamma);
      T(l) = tj ;
    }
    T.clamp(1e-8,1e6);
    sumT    = arma::accu(T);
    U       = arma::randg(J,arma::distr_param(sumT, 1.0));
    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      Beta = T / sumT ;

      Pisum  = arma::sum(Pi, 0);
      Pi_out = Pi;
      for ( j = 0; j < J; ++j) {
        if (Pisum(j) != 1.0) {
          Pi_out.col(j) /= Pisum(j);
        }
      }

      return_list[ss] = Rcpp::List::create(
        Rcpp::Named("Imputation") = augY,
        Rcpp::Named("Z")          = Z,
        Rcpp::Named("Phi")        = wrap_param_cluster(PHI),
        Rcpp::Named("Pi")         = Pi_out,
        Rcpp::Named("Beta")       = Beta,
        // Rcpp::Named("U")          = U,
        // Rcpp::Named("T")          = T,
        Rcpp::Named("alpha")      = alpha);
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return return_list;
}

//' Hierarchical BNP imputation via MCMC with fixed alpha
//'
//' Runs the same MCMC algorithm as \code{sBNPi}, but keeps the missingness
//' dependence parameter \code{alpha} fixed throughout the sampler instead of
//' updating it at each iteration.
//'
//' @param data Numeric matrix (\eqn{n} x \eqn{q}) of observed data. Missing values must be
//'   encoded as not-available, i.e. \code{NA}.
//' @param group Integer vector of length \eqn{n} indicating group membership for
//'   each observation. Values must be in \eqn{\{0, \ldots, J-1\}}.
//' @param alpha Fixed scalar parameter controlling the missingness structure.
//' @param c0,alpha0 Hyperparameters of the base-measure component for the
//'   reference pattern.
//' @param Sigma0,mu0,kappa0,nu0 Hyperparameters of the base-measure component
//'   for the complete-data parameters \eqn{\mu} and \eqn{\Sigma}.
//' @param gamma0,eta0 Positive scalars. Hyperparameters for the global mixture
//'   weights.
//' @param L Integer. Number of mixture components, i.e. finite truncation level.
//' @param sample Integer. Number of MCMC samples to store.
//' @param burn Integer. Number of burn-in iterations.
//' @param thinning Integer. Thinning interval.
//' @param cluster_init Initialization strategy for the cluster allocations. One
//'   of \code{"random"}, \code{"cluster"}, \code{"singleton"}, or \code{"manual"}.
//' @param allocation_init \code{NULL} or an integer vector used when
//'   \code{cluster_init = "manual"}.
//' @param approx_tilted_gamma Integer. Number of splits controlling the
//'   tilted-gamma approximation.
//' @param maxiter_tilted_gamma Integer. Maximum number of iterations for the
//'   tilted-gamma mode solver.
//'
//' @return A list of length \code{sample}. Each element contains:
//' \describe{
//'   \item{\code{Imputation}}{Input data matrix with \code{NA} values replaced by imputed values.}
//'   \item{\code{Z}}{Cluster allocation vector of length \eqn{n}.}
//'   \item{\code{Phi}}{List of cluster-specific parameters. Each element contains \code{c}, \code{mu}, and \code{Sigma}.}
//'   \item{\code{Pi}}{An \eqn{L} x \eqn{J} matrix of group-specific mixture weights.}
//'   \item{\code{Beta}}{Vector of length \eqn{L} representing global mixture weights.}
//' }
//'
//' @seealso \code{\link{sBNPi}}
//'
//' @examples
//' \dontrun{
//' set.seed(456)
//'
//' # Same simulated setup as in sBNPi()
//' n <- 120; q <- 3; J <- 2
//' group <- sample(0:(J - 1), n, replace = TRUE)
//' z <- sample(1:2, n, replace = TRUE)
//' Y <- t(sapply(z, function(k) {
//'   if (k == 1) rnorm(q, mean = c(-2, -2, 0)) else rnorm(q, mean = c(2, 2, 0))
//' }))
//' Y[sample(length(Y), round(0.20 * length(Y)))] <- NA
//'
//' # Run with alpha held fixed (e.g. at 0.25)
//' fit <- sBNPi_alpha(data    = Y,
//'                    group   = group,
//'                    alpha   = 0.25,
//'                    c0      = rep(0L, q), alpha0 = 0.1,
//'                    Sigma0  = diag(q),    mu0    = rep(0, q),
//'                    kappa0  = 1,          nu0    = q + 2,
//'                    gamma0  = 1,          eta0   = 1,
//'                    L       = 10,
//'                    sample  = 200, burn = 100, thinning = 1)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List sBNPi_alpha( const arma::mat  & data,
                        const arma::ivec & group, // index j
                        const double alpha,
                        const arma::ivec  & c0, const double & alpha0,
                        const arma::mat  & Sigma0,   const arma::vec & mu0,
                        const double kappa0,         const double nu0,
                        const double gamma0, const double eta0, // b0 in Snigdha Das, Yabo Niu, Yang Ni, Bani K. Mallick, Debdeep Pati (2025).
                        const int L = 10,
                        const unsigned int sample = 1000, const unsigned int burn = 0, const unsigned int thinning = 1,
                        const std::string cluster_init = "random",
                        const Rcpp::Nullable<arma::ivec> allocation_init = R_NilValue,
                        const int approx_tilted_gamma  = 1, const int maxiter_tilted_gamma = 1e3){

  // Index for cycles, useful indeces,
  arma::ivec r_i;
  arma::vec  y_i;
  arma::vec  Pi_j;

  arma::ivec unique =  arma::unique(group);
  const int J  = unique.n_elem ;
  const int q  = data.n_cols  ;
  const int NN = data.n_rows ;
  const int mL = L-1 ;

  int i, ii, j, l;
  // YC:  Augmented data
  // R:   Missing indicator
  // Dim: J x mL
  arma::mat RR( NN, q, arma::fill::ones) ;
  arma::mat  augY = data ;
  arma::uvec idx_mis;
  arma::uvec idx_obs;
  for( int var = 0 ; var < q; ++var){
    // iter
    arma::colvec     ycol = augY.col(var);
    arma::colvec     rcol = RR.col(var);
    // idx
    arma::uvec idx_mis = arma::find_nonfinite(ycol);
    arma::uvec idx_obs = arma::find_finite(ycol);
    if(!idx_mis.is_empty() && !idx_obs.is_empty()){
      double col_mean = arma::mean(ycol(idx_obs));
      ycol(idx_mis).fill(col_mean);
      rcol(idx_mis).zeros();
    }
    augY.col(var) = ycol;
    RR.col(var)   = rcol;
  }
  // Iterator_for_missing: std::vector to cycle with information for data augmentation step
  std::vector<missing> iterator_for_missing ;
  missing mis_map_it ;
  for( i = 0 ; i < NN; ++ i){
    mis_map_it = missing(data.row(i).as_col(), i) ;
    if( mis_map_it.n_mis != 0 ){
      iterator_for_missing.push_back( mis_map_it ) ;
    }
  }

  // Z: Allocations
  //    arma::vec of length NN with values from 1 to L
  //    the position is the same of `group` (j)
  arma::ivec Z;

  if (cluster_init == "random") {
    Z = arma::randi<arma::ivec>(NN, arma::distr_param(0,mL));
  } else if (cluster_init == "cluster") {
    // TO DO
    Rcpp::stop("cluster implementation is to do");
  } else if (cluster_init == "singleton") {
    if( NN <= L ){
      Z = arma::regspace<arma::ivec>(0, NN - 1);
    }else{
      Rcpp::stop("Error: cluster_init = 'singleton' requires L greater than or equal to the sample size.");
    }
  } else if (cluster_init == "manual") {
    if (allocation_init.isNull()) {
      Rcpp::stop("Error: init = 'manual' requires non NULL allocation_init, you must provide it.");
    }
    Z = Rcpp::as<arma::ivec>(allocation_init);
    if ((int) Z.n_elem != NN) {
      Rcpp::stop("Error: init_rho must have length equal to nrow(data).");
    }
    // TO DO: add controll for values in \{0, ..., L\}
  } else {
    Rcpp::stop("Invalid init. Use one of: 'random', 'cluster', 'singleton', 'manual'.");
  }

  // PHI: Parameters for finite approximation
  //      length = J*L
  std::vector<param_cluster> PHI(L);

  arma::mat Sigma_init;
  arma::colvec mu_init;
  arma::ivec   c_init;
  Sigma_init.eye(q,q);

  for( int l = 0; l < L ; ++l){
    param_cluster & par = PHI[l];
    c_init  = arma::randi(q, arma::distr_param(0,1));
    mu_init = arma::randn(q) ;
    par = param_cluster( c_init, mu_init,  Sigma_init);
  }

  // create Pi (vector of group specific probabilities)
  arma::mat Pi(L,J);
  Pi.fill(1.0/L);

  // create Sufficient Statistics to update Pi
  arma::mat Pi_ss(L,J, arma::fill::zeros) ;
  for( int ii = 0; ii < NN ;  ++ii ){
    l = Z(ii) ;
    j = group(ii) ;
    Pi_ss(l,j) +=1 ;
  }
  arma::mat    Pi_par(L,J) ;
  arma::colvec Dir_par(L) ;

  // arma::colvec T, length = L
  arma::colvec    T(L, arma::fill::ones);
  double sumT = arma::accu(T);
  arma::colvec Beta = T / sumT ;

  // arma::colvec U (arma vector, length = J
  arma::colvec U(J, arma::fill::ones);
  double ulogsum = 0.0 ;

  arma::colvec B = eta0 - arma::sum(arma::log(Pi), 1) - ulogsum ;

  const double A = gamma0/L;
  double Bk, tj;

  // MCMC Algorithm
  Rcpp::List return_list(sample);
  arma::rowvec Pisum  ; // somme per colonna
  arma::mat    Pi_out ;
  const unsigned int S = burn + thinning * sample;
  unsigned ss = 0;

  const arma::mat Sigma0Inv = arma::inv_sympd(Sigma0);

  Rcpp::Rcout << "\rInzializing starting values and constant for MCMC iterations. (Done)" << std::endl ;
  MyTimePoint t0 = myClock::now();

  for( unsigned int s = 0; s<S; ++s){
    Rcpp::checkUserInterrupt();
    catIter(s, S, t0) ;

    // 1: Missing Imputation
    for ( auto it  = iterator_for_missing.begin();
          it != iterator_for_missing.end();
          ++it) {
      // mapping with index
      ii  = (it->subject_idx) ;
      j   = group(ii);
      l   = Z(ii)    ;
      // c++ references
      arma::subview_row<double> yi_it = augY.row(ii);
      param_cluster & par_it = PHI[l];
      // Call update
      missing_data_augmentation( yi_it, (it->mis), (it->obs), par_it.mu, par_it.Sigma ) ;
    }

    // 2: Update Z
    for(ii=0; ii < NN; ++ii){
      // mapping with idex
      j    = group(ii);
      r_i  = arma::conv_to<arma::ivec>::from( RR.row(ii).t() );;
      y_i  = augY.row(ii).t();
      Pi_j = Pi.col(j);
      int & z_ji = Z(ii) ;
      // Call update
      Pi_ss( z_ji, j) -= 1 ;
      z_ji = update_cluster_membership(Pi_j, y_i, r_i, PHI, alpha);
      Pi_ss( z_ji, j) += 1 ;
    }

    // 3: Update Phi
    for(l=0; l < L; ++l){
      // References
      param_cluster &  par_it      = PHI[l];
      arma::ivec    &  c_it        = par_it.c ;
      arma::vec     &  mu_it       = par_it.mu ;
      arma::mat     &  Sigma_it    = par_it.Sigma;
      arma::mat     &  SigmaInv_it = par_it.SigmaInv;

      arma::uvec    idx_l = arma::find(Z==l);
      arma::mat Yc_submat = augY.rows(idx_l);
      arma::mat RR_submat = RR.rows(idx_l);

      update_phi( Yc_submat, RR_submat,
                  c_it, mu_it, Sigma_it, SigmaInv_it,
                  alpha, c0, alpha0, Sigma0Inv, Sigma0, mu0, nu0, kappa0);
    }

    // 3: For (j in 1,...,J) update riga J di Matrice Pi (dirichlet)
    for(j=0; j<J ; ++j){
      Dir_par   = Pi_ss.col(j) + T ;
      Pi.col(j) = cpp_rdirichlet_1(Dir_par);
    }
    Pi.clamp(1e-6, 1-1e-6) ;
    ulogsum = arma::accu(arma::log(U)) ;
    B       = eta0 - arma::sum( arma::log(Pi), 1) - ulogsum ;

    // 4: For (k in 1,...,L) update elemento t_j del vettore T
    for( l = 0 ; l < L ; ++l ){
      Bk = std::min( B(l), 1e8 );
      tj = cpp_rtilted_gamma_1(  A, Bk, J, approx_tilted_gamma,  maxiter_tilted_gamma);
      T(l) = tj ;
    }
    T.clamp(1e-8,1e6);
    sumT    = arma::accu(T);
    U       = arma::randg(J,arma::distr_param(sumT, 1.0));
    // Store
    if( ( (s+1) > burn ) & ((s+1-burn) % thinning == 0)){
      Beta = T / sumT ;

      Pisum  = arma::sum(Pi, 0);
      Pi_out = Pi;
      for ( j = 0; j < J; ++j) {
        if (Pisum(j) != 1.0) {
          Pi_out.col(j) /= Pisum(j);
        }
      }

      return_list[ss] = Rcpp::List::create(
        Rcpp::Named("Imputation") = augY,
        Rcpp::Named("Z")          = Z,
        Rcpp::Named("Phi")        = wrap_param_cluster(PHI),
        Rcpp::Named("Pi")         = Pi_out,
        Rcpp::Named("Beta")       = Beta
        // Rcpp::Named("U")          = U,
        // Rcpp::Named("T")          = T,
        );
      ss += 1 ;
    }
  }
  catIter(S, S, t0) ;
  return return_list;
}
