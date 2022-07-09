"""
    update_sspace_Q_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Update `sspace.Q` from `params`.
"""
function update_sspace_Q_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)
    if is_llt
        sspace.Q[1, 1] = sspace.Q[2, 2]*params[1];
        sspace.Q[3, 3] = sspace.Q[2, 2]*params[2];
    else
        sspace.Q[2, 2] = sspace.Q[1, 1]*params[1];
    end
end

"""
    update_sspace_C_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Update `sspace.C` from `params`.
"""
function update_sspace_C_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)    
    sspace.C[3, 3:end] = params[2+is_llt:end];
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
    if (sum(isnan.(sspace.C)) == 0) && (sum(isinf.(sspace.C)) == 0)
        is_cycle_non_problematic = maximum(abs.(eigvals(sspace.C[3:end, 3:end]))) <= 0.98;
    else
        is_cycle_non_problematic = false;
    end

    # Determine whether Q is problematic
    if (sum(isnan.(sspace.Q)) == 0) && (sum(isinf.(sspace.Q)) == 0)
        is_Q_non_problematic = true;
    else
        is_Q_non_problematic = false;
    end

    # Determine whether P0 is problematic
    if (sum(isnan.(sspace.P0)) == 0) && (sum(isinf.(sspace.P0)) == 0)
        is_P0_non_problematic = minimum(abs.(eigvals(sspace.P0))) >= 1e-8;
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
    call_fmin!(constrained_params::Vector{Float64}, tuple_fmin_args::Tuple)

APIs to call `fmin!` with Tuple parameters.
"""
call_fmin!(constrained_params::Vector{Float64}, tuple_fmin_args::Tuple) = fmin!(constrained_params, tuple_fmin_args...);

"""
    call_reparametrised_fmin!(params::Vector{Float64}, tuple_args::Tuple)

APIs to call `fmin!` after transforming the candidate parameters so that their support is unbounded.
"""
function call_reparametrised_fmin!(params::Vector{Float64}, tuple_args::Tuple)

    params_lb = tuple_args[1];
    params_ub = tuple_args[2];
    tuple_fmin_args = tuple_args[3:end];

    # Unconstrained -> constrained parameters
    constrained_params = similar(params);
    for i in axes(constrained_params, 1)
        constrained_params[i] = get_bounded_logit(params[i], params_lb[i], params_ub[i]);
    end
    
    return call_fmin!(constrained_params, tuple_fmin_args);
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
    
    # Local linear trend or Kitagawa second order trend (special case of the local linear trend)
    else
        C_trend = [1.0 1.0; 0.0 1.0];
    end

    # Stationary dynamics
    C = cat(dims=[1,2], C_trend, companion_form([0.9 zeros(1, lags-1)], extended=false));

    # Remaining transition equation coefficients
    D = zeros(2+lags, 2+is_llt);
    
    # Drift-less random walk
    if is_rw_trend
        D[1,1] = 1.0;
        D[3,2] = 1.0;

    # Local linear trend
    elseif is_llt
        D[1,1] = 1.0;
        D[2,2] = 1.0;
        D[3,3] = 1.0;

    # Kitagawa second order trend (special case of the local linear trend)
    else
        D[2,1] = 1.0;
        D[3,2] = 1.0;
    end

    # Covariance matrix of the transition innovations
    Q = Symmetric(1.0*Matrix(I, 2+is_llt, 2+is_llt));

    # Fix the variance of the drift (or trend in the case in which is_rw_trend == true)
    Q[1+is_llt, 1+is_llt] = 1e-4;

    # Initial conditions (mean)
    X0 = zeros(2+lags);

    # Initial conditions (covariance)
    C_cycle = C[3:end, 3:end];
    DQD_cycle = Symmetric(cat(dims=[1,2], Q[2+is_llt, 2+is_llt], zeros(lags-1, lags-1)));
    P0_cycle = solve_discrete_lyapunov(C_cycle, DQD_cycle).data;
    P0_trend = Symmetric(Inf*Matrix(I, 2, 2));
    oom_maximum_P0_cycle = floor(Int, log10(maximum(P0_cycle)));  # order of magnitude of the largest entry in P0_cycle
    P0_trend[isinf.(P0_trend)] .= 10.0^(oom_maximum_P0_cycle+4); # non-stationary components (variances)
    P0 = Symmetric(cat(dims=[1,2], P0_trend, P0_cycle));
    
    # Set KalmanSettings
    sspace = KalmanSettings(Array(data'), B, R, C, D, Q, X0, P0, compute_loglik=true);

    #=
    Initial values / bounds for the parameters

    When is_llt == false:
    - sigma_{cycle}^2 / sigma_{drift}^2
    - AR(p) parameters

    When is_llt == true:
    - sigma_{trend}^2 / sigma_{drift}^2
    - sigma_{cycle}^2 / sigma_{trend}^2
    - AR(p) parameters

    Note 1: sigma_{drift}^2 is fixed to 1e-4 during the initialisation
    Note 2: in the case in which is_rw_trend == true, sigma_{drift}^2 denotes the variance of the trend.
    =#

    params_0  = [1e3; 0.90; zeros(lags-1)];
    params_lb = [1e2; -2*ones(lags)];
    params_ub = [1e4; +2*ones(lags)];

    if is_llt
        pushfirst!(params_0,  1e-3);
        pushfirst!(params_lb, 1e-4);
        pushfirst!(params_ub, 1e-2);
    end
    
    # Maximum likelihood
    tuple_fmin_args = (is_llt, sspace);
    prob = OptimizationFunction(call_fmin!) #, Optimization.AutoForwardDiff());
    prob = OptimizationProblem(prob, params_0, tuple_fmin_args, lb=params_lb, ub=params_ub);
    res_optim = solve(prob, NLopt.LN_NELDERMEAD(), abstol=0.0, reltol=1e-3);
    minimizer_bounded = res_optim.u;
    
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
