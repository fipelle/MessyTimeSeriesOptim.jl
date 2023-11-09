"""
"""
function initial_detrending_step_1()
    
    # Get initial state-space parameters and relevant coordinates
    B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0 = initial_sspace_structure(Y_trimmed, estim);

    # Simplified structure for step 1 of the initialisation
    coordinates_last_idio = findall(B[1, :] .!= 0.0)[end]-1;

    # Measurement equation
    B = B[:, 1:coordinates_last_idio];
    coordinates_free_params_B = coordinates_free_params_B[:, 1:coordinates_last_idio];

    # Transition equation
    C = C[1:coordinates_last_idio, 1:coordinates_last_idio];
    D = D[1:coordinates_last_idio, 1:(estim.n_trends + n_trimmed)]
    Q = Symmetric(Q[1:(estim.n_trends + n_trimmed), 1:(estim.n_trends + n_trimmed)]);

    # Update coefficients for the cycles where needed
    C[C .== 0.1] .*= 9;
    
    # Initial conditions
    X0 = X0[1:coordinates_last_idio];
    P0 = Symmetric(P0[1:coordinates_last_idio, 1:coordinates_last_idio]);
    coordinates_free_params_P0 = coordinates_free_params_P0[1:coordinates_last_idio, 1:coordinates_last_idio];

    # Set KalmanSettings
    sspace = KalmanSettings(Y_trimmed, B, R, C, D, Q, X0, P0, compute_loglik=true);

    # Initial guess for the parameters
    params_0 = vcat(
        1e4*ones(1+n_trimmed),
        1e-4*ones(estim.n_trends),
    );
    params_lb = vcat(1e+2*ones(1+n_trimmed), 1e-4*ones(estim.n_trends));
    params_ub = vcat(1e+6*ones(1+n_trimmed), 1e+4*ones(estim.n_trends));
    
    # Maximum likelihood
    tuple_fmin_args = (sspace, coordinates_free_params_B, coordinates_free_params_P0);
    prob = OptimizationFunction(call_fmin!)
    prob = OptimizationProblem(prob, params_0, tuple_fmin_args, lb=params_lb, ub=params_ub);
    res_optim = solve(prob, NLopt.LN_SBPLX, abstol=1e-4, reltol=1e-2);
    
    # Return minimizer
    return res_optim.u;
end

"""
"""
function initial_detrending_step_2()
    
    # Get initial state-space parameters and relevant coordinates
    B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0 = initial_sspace_structure(Y_trimmed, estim);

    # Set KalmanSettings
    sspace = KalmanSettings(Y_trimmed, B, R, C, D, Q, X0, P0, compute_loglik=true);

    # Recover initial guess from step 1
    params_0 = initial_detrending_step_1();

    # TBA PCA
    # -> loading matrix in the correct position within `B`

    # Update `params_0`
    params_0 = vcat(B[coordinates_free_params_B], params_0);
    params_lb = vcat(B[coordinates_free_params_B]/10, 1e+2*ones(1+n_trimmed), 1e-4*ones(estim.n_trends));
    params_ub = vcat(B[coordinates_free_params_B]*10, 1e+6*ones(1+n_trimmed), 1e+4*ones(estim.n_trends));
end