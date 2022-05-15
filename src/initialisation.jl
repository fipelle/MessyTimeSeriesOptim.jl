"""
    update_sspace_Q_from_params!(params::Vector{Float64}, sspace::KalmanSettings)

Update `sspace.Q` from `params`. The variance of the trend is smaller by construction and by a factor of `params[2]`.
"""
function update_sspace_Q_from_params!(params::Vector{Float64}, sspace::KalmanSettings)
    sspace.Q[1,1] = params[1];
    sspace.Q[2,2] = params[1]*params[2];
end

"""
    update_sspace_C_from_params!(params::Vector{Float64}, sspace::KalmanSettings)

Update `sspace.C` from `params`.
"""
function update_sspace_C_from_params!(params::Vector{Float64}, sspace::KalmanSettings)
    sspace.C[3, 3:end] = params[3:end];
end

"""
    update_sspace_P0_from_params!(sspace::KalmanSettings)

Update `sspace.P0` from `params`.
"""
function update_sspace_P0_from_params!(sspace::KalmanSettings)
    C_cycle = sspace.C[3:end, 3:end];
    DQD_cycle = Symmetric((sspace.D*sspace.Q*sspace.D')[3:end, 3:end]);
    sspace.P0.data[3:end, 3:end] = solve_discrete_lyapunov(C_cycle, DQD_cycle).data;
end

"""
    initial_univariate_decomposition_fmin!(params::Vector{Float64}, sspace::KalmanSettings)

Return -1 times the log-likelihood function (or a large number if the cycle is not causal).
"""
function initial_univariate_decomposition_fmin!(params::Vector{Float64}, sspace::KalmanSettings)
    
    # Update sspace accordingly
    update_sspace_Q_from_params!(params, sspace);
    update_sspace_C_from_params!(params, sspace);
    
    # Make sure that the cycle is causal
    is_cycle_causal = maximum(abs.(eigvals(sspace.C[3:end, 3:end]))) <= 0.98;

    # Return fmin
    if is_cycle_causal

        # Update initial conditions
        update_sspace_P0_from_params!(sspace);
        
        # Run kalman filter
        status = kfilter_full_sample(sspace);

        # Return fmin
        return -status.loglik;
    
    else
        return 1/eps();
    end
end

"""
    initial_univariate_decomposition(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool)

This function returns an initial estimate of the non-stationary and stationary components of each series.
In doing so, it provides a rough starting point for the ECM algorithm.
"""
function initial_univariate_decomposition(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool)
    
    # Measurement equation coefficients
    B = [1.0 0.0 1.0 zeros(1, lags-1)];
    R = ε*I;

    # Non-stationary dynamics
    if is_rw_trend
        # Note: this is not the most compressed representation, but simplifies the remaining part of this function without significantly compromising the run time
        C_trend = [1.0 0.0; 1.0 0.0];
        P0_trend = Symmetric(1e3*Matrix(I, 2, 2));
    else
        C_trend = [2.0 -1.0; 1.0 0.0];
        P0_trend = 1e3*ones(2,2);
    end

    # Stationary dynamics
    C = cat(dims=[1,2], C_trend, companion_form([0.9 zeros(1, lags-1)], extended=false));

    # Remaining transition coefficients
    D = zeros(2+lags, 2);
    D[1,1] = 1.0;
    D[3,2] = 1.0;
    Q = Symmetric(1.0*Matrix(I, 2, 2));
    Q[1,1] = 1e-4;
    Q[2,2] = 0.1;

    # Initial conditions (mean)
    X0 = zeros(2+lags);
    
    # Initial conditions (covariance)
    C_cycle = C[3:end, 3:end];
    DQD_cycle = Symmetric(cat(dims=[1,2], Q[2,2], zeros(lags-1, lags-1)));
    P0 = Symmetric(cat(dims=[1,2], P0_trend, solve_discrete_lyapunov(C_cycle, DQD_cycle).data));
    
    # Set KalmanSettings
    sspace = KalmanSettings(Array(data'), B, R, C, D, Q, X0, P0, compute_loglik=true);

    # Maximum likelihood
    params_0  = [1e-2; 1e2; 0.90; zeros(lags-1)];
    params_lb = [1e-4; 1.0; -10*ones(lags)];
    params_ub = [1.0;  1e4; +10*ones(lags)];
    res = optimize(params -> initial_univariate_decomposition_fmin!(params, sspace), params_lb, params_ub, params_0, SAMIN(rt=0.90, f_tol=0.0, x_tol=1e-6), Optim.Options(iterations = 10^6));
    
    # Update sspace accordingly
    update_sspace_Q_from_params!(res.minimizer, sspace);
    update_sspace_C_from_params!(res.minimizer, sspace);
    update_sspace_P0_from_params!(sspace);

    # Retrieve optimal states
    status = kfilter_full_sample(sspace);
    smoothed_states, _ = ksmoother(sspace, status);
    smoothed_states_matrix = hcat(smoothed_states...);
    trend = smoothed_states_matrix[1, :];
    cycle = smoothed_states_matrix[3, :];
    trend_variance = sspace.Q[1, 1];
    
    # Return output
    return trend, cycle, trend_variance;
end