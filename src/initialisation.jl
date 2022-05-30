"""
    update_sspace_Q_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Update `sspace.Q` from `params`. The variance of the trend is smaller by construction and by a factor of `params[2]`.
"""
function update_sspace_Q_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)
    sspace.Q[1, 1] = params[1];

    if is_llt
        sspace.Q[2, 2] = params[2];
        sspace.Q[3, 3] = 0.5*(params[1]+params[2])*params[3];
    else
        sspace.Q[2, 2] = params[1]*params[2];
    end
end

"""
    update_sspace_C_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Update `sspace.C` from `params`.
"""
function update_sspace_C_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)
    
    # Update `params` to enhance mixing
    scaling_factor = max(1, sum(abs.(params[3+is_llt:end])));
    for i=3+is_llt:length(params)
        params[i] /= scaling_factor; # this improves mixing since any resulting eigenvalue of the companion form of the cycle will be <= 1 in absolute value
        params[i] *= 0.98;           # while this should ensure that the eigenvalues are <= 0.98 in absolute value, numerical errors may lead to problematic cases <- these are handled below
    end

    # Update `sspace.C`
    sspace.C[3, 3:end] = params[3+is_llt:end];
end

"""
    update_sspace_DQD_and_P0_from_params!(sspace::KalmanSettings)

Update `sspace.DQD` and `sspace.P0` from `params`.
"""
function update_sspace_DQD_and_P0_from_params!(sspace::KalmanSettings)
    
    # Update `sspace.DQD`
    sspace.DQD.data .= Symmetric(sspace.D*sspace.Q*sspace.D').data;

    # Update `sspace.P0`
    C_cycle = sspace.C[3:end, 3:end];
    DQD_cycle = Symmetric(sspace.DQD[3:end, 3:end]);
    sspace.P0.data[3:end, 3:end] = solve_discrete_lyapunov(C_cycle, DQD_cycle).data;
end

"""
    fmin!(constrained_params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Return -1 times the log-likelihood function (or a large number if the cycle is not causal).
"""
function fmin!(constrained_params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

    # Update sspace accordingly
    update_sspace_Q_from_params!(constrained_params, is_llt, sspace);
    update_sspace_C_from_params!(constrained_params, is_llt, sspace);

    # Determine whether the cycle is problematic
    if sum(isinf.(sspace.C)) == 0
        is_cycle_non_problematic = maximum(abs.(eigvals(sspace.C[3:end, 3:end]))) <= 0.98;
    else
        is_cycle_non_problematic = false;
    end

    # Determine whether Q is problematic
    if sum(isinf.(sspace.Q)) == 0
        is_Q_non_problematic = true;
    else
        is_Q_non_problematic = false;
    end    

    # Determine whether P0 is problematic
    if (sum(isnan.(sspace.P0)) == 0) && (sum(isinf.(sspace.P0)) == 0)
        is_P0_non_problematic = minimum(abs.(eigvals(sspace.P0))) >= 1e-4;
    else
        is_P0_non_problematic = false;
    end
    
    # Regular run
    if is_cycle_non_problematic && is_Q_non_problematic && is_P0_non_problematic
        
        # Update initial conditions
        update_sspace_DQD_and_P0_from_params!(sspace);
        
        # Run kalman filter and return -loglik
        try            
            status = kfilter_full_sample(sspace);
            return -status.loglik;

        # Problematic run
        catch kf_run_error
            if isa(kf_run_error, DomainError)
                return 1/eps();
            else
                println("B")
                throw(kf_run_error);
            end
        end
    
    # Problematic run
    else
        return 1/eps();
    end
end

"""
    fmin_logit_transformation!(params::Vector{Float64}, params_lb::Vector{Float64}, params_ub::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Trasform params to have a bounded support and then run fmin!(...).
"""
function fmin_logit_transformation!(params::Vector{Float64}, params_lb::Vector{Float64}, params_ub::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

    # Unconstrained -> constrained parameters
    constrained_params = copy(params);
    for i in axes(constrained_params, 1)
        constrained_params[i] = get_bounded_logit(constrained_params[i], params_lb[i], params_ub[i]);
    end
        
    # Return -1 times the log-likelihood function
    return fmin!(constrained_params, is_llt, sspace);
end

"""
    initial_univariate_decomposition(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool, is_llt::Bool)

This function returns an initial estimate of the non-stationary and stationary components of each series. In doing so, it provides a rough starting point for the ECM algorithm.

# Note
- If both `is_rw_trend` and `is_llt` are false this function models the trend as in Kitagawa and Gersch (1996, ch. 8).
"""
function initial_univariate_decomposition(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool, is_llt::Bool)
    
    if is_rw_trend && is_llt
        error("Either is_rw_trend or is_llt can be true");
    end
    
    # Measurement equation coefficients
    B = [1.0 0.0 1.0 zeros(1, lags-1)];
    R = ε*I;
    
    # Drift-less random walk
    if is_rw_trend
        C_trend = [1.0 0.0; 1.0 0.0]; # this is not the most compressed representation, but simplifies the remaining part of this function without significantly compromising the run time
        P0_trend = Symmetric(1e3*Matrix(I, 2, 2));
    
    # Local linear trend
    elseif is_llt
        C_trend = [1.0 1.0; 0.0 1.0];
        P0_trend = 1e3*ones(2,2);
    
    # Kitagawa second order trend (special case of the local linear trend)
    else
        C_trend = [2.0 -1.0; 1.0 0.0];
        P0_trend = 1e3*ones(2,2);
    end

    # Stationary dynamics
    C = cat(dims=[1,2], C_trend, companion_form([0.9 zeros(1, lags-1)], extended=false));

    # Remaining transition equation coefficients
    D = zeros(2+lags, 2+is_llt);
    D[1,1] = 1.0;
    if is_rw_trend || ~is_llt
        D[3,2] = 1.0;
    else
        D[2,2] = 1.0;
        D[3,3] = 1.0;
    end

    # Covariance matrix of the transition innovations
    Q = Symmetric(1.0*Matrix(I, 2+is_llt, 2+is_llt));

    # Initial conditions (mean)
    X0 = zeros(2+lags);
    
    # Initial conditions (covariance)
    C_cycle = C[3:end, 3:end];
    DQD_cycle = Symmetric(cat(dims=[1,2], Q[2+is_llt, 2+is_llt], zeros(lags-1, lags-1)));
    P0 = Symmetric(cat(dims=[1,2], P0_trend, solve_discrete_lyapunov(C_cycle, DQD_cycle).data));

    # Set KalmanSettings
    sspace = KalmanSettings(Array(data'), B, R, C, D, Q, X0, P0, compute_loglik=true);

    # Initial values / bounds for the parameters
    if is_rw_trend || ~is_llt
        params_0  = [1e-2; 1e2; 0.90; zeros(lags-1)];
        params_lb = [1e-4; 1e1; -2*ones(lags)];
        params_ub = [1.00; 1e3; +2*ones(lags)];
    else
        params_0  = [1e-8; 1e-2; 1e2; 0.90; zeros(lags-1)];
        params_lb = [0.00; 1e-4; 1e1; -2*ones(lags)];
        params_ub = [1.00; 1.00; 1e3; +2*ones(lags)];
    end

    # Maximum likelihood
    params_0_unb = [get_unbounded_logit(params_0[i], params_lb[i], params_ub[i]) for i in axes(params_0, 1)];
    res_optim = Optim.optimize(params -> fmin_logit_transformation!(params, params_lb, params_ub, is_llt, sspace), params_0_unb, Newton(), Optim.Options(f_reltol=1e-6, x_reltol=1e-6, iterations=10^6));

    # Minimiser with bounded support
    minimizer_bounded = [get_bounded_logit(res_optim.minimizer[i], params_lb[i], params_ub[i]) for i in axes(res_optim.minimizer, 1)];

    # Update sspace accordingly
    update_sspace_Q_from_params!(minimizer_bounded, is_llt, sspace);
    update_sspace_C_from_params!(minimizer_bounded, is_llt, sspace);
    update_sspace_DQD_and_P0_from_params!(sspace);

    # Retrieve optimal states
    status = kfilter_full_sample(sspace);
    smoothed_states, _ = ksmoother(sspace, status);
    smoothed_states_matrix = hcat(smoothed_states...);
    trend = smoothed_states_matrix[1, :];
    drift_or_trend_lagged = smoothed_states_matrix[2, :];
    cycle = smoothed_states_matrix[3, :];
    
    # Return output
    return trend, drift_or_trend_lagged, cycle, status;
end

"""
    initial_univariate_decomposition_kitagawa(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool)

This function returns an initial estimate of the non-stationary and stationary components of each series. In doing so, it provides a rough starting point for the ECM algorithm.

If `is_rw_trend` is false this function models the trend as in Kitagawa and Gersch (1996, ch. 8).
"""
function initial_univariate_decomposition_kitagawa(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool)
    trend, _, cycle, _ = initial_univariate_decomposition(data, lags, ε, is_rw_trend, false);
    return trend, cycle;
end

"""
    initial_univariate_decomposition_llt(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool)

This function returns an initial estimate of the non-stationary and stationary components of each series. In doing so, it provides a rough starting point for the ECM algorithm.

If `is_rw_trend` is false this function models the trend as a local linear trend.
"""
function initial_univariate_decomposition_llt(data::JVector{Float64}, lags::Int64, ε::Float64, is_rw_trend::Bool)
    trend, drift, cycle, _ = initial_univariate_decomposition(data, lags, ε, is_rw_trend, true);
    return trend, drift, cycle;
end
