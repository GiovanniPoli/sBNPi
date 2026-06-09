if(FALSE){

  q=6
  # model hyperparameters
  hypar = list( a0 = 1, b0 = 1, c0 = rep(1, q), alpha0   = 1/2,
              Sigma0   = diag(q), mu0 = rep(0,q), kappa0 = 1, nu0  = q+2,
              gamma0   = 1, eta0 = 1,
              L        = 50,
              sample   = 1000, burn = 10000, thinning = 10,
              approx_tilted_gamma  = 1, maxiter_tilted_gamma = 1000 )

  eps = 1e-2
  alpha_vec = c(eps, seq(0.1, 0.2, 0.1), 0.4, 0.5-eps)
  rho_vec   = seq(0.1, 0.9, length.out=5 )
  alpha_rho_comb = expand.grid(alpha=alpha_vec, rho=rho_vec)

  alpha = alpha_rho_comb[1, 1]
  rho   = alpha_rho_comb[1, 2]

  it = list(alpha = alpha, rho = rho, seed = 1)

  t = it$seed

  st<-Sys.time()

  data_gen = sBNPi::gen_data_S2( seed = it$seed, alpha = it$alpha, rho = it$rho )

  Y_mat_complete = data_gen$sim_truth
  Y_mat_obs      = data_gen$sim_data
  RR_mat         = data_gen$RR_mat
  group          = data_gen$sim_pat

  model_fit = sBNPi(       data     = Y_mat_obs,
                           group    = group,
                           a0       = hypar$a0,
                           b0       = hypar$b0,
                           c0       = hypar$c0,
                           alpha0   = hypar$alpha0,
                           Sigma0   = hypar$Sigma0,
                           mu0      = hypar$mu0,
                           kappa0   = hypar$kappa0,
                           nu0      = hypar$nu0,
                           gamma0   = hypar$gamma0,
                           eta0     = hypar$eta0,
                           L        = hypar$L,
                           sample   = hypar$sample,
                           burn     = hypar$burn,
                           thinning = hypar$thinning,
                           cluster_init         = "random",
                           approx_tilted_gamma  = hypar$approx_tilted_gamma,
                           maxiter_tilted_gamma = hypar$maxiter_tilted_gamma)
}
