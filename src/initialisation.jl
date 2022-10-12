"""
    update_sspace_Q_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Update `sspace.Q` from `params`.

# Notes on the parameters

When is_llt == false:
- sigma_{drift}^2
- sigma_{cycle}^2 / sigma_{drift}^2
- AR(p) parameters

When is_llt == true:
- sigma_{drift}^2
- sigma_{trend}^2 / sigma_{drift}^2
- sigma_{cycle}^2 / sigma_{drift}^2
- AR(p) parameters

In the case in which is_rw_trend == true, sigma_{drift}^2 denotes the variance of the trend.
"""
function update_sspace_Q_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)
    sspace.Q[1, 1] = params[1];
    sspace.Q[2, 2] = params[1]*params[2];
    if is_llt
        sspace.Q[3, 3] = params[1]*params[3];
    end
end

"""
    update_sspace_C_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)

Update `sspace.C` from `params`.

# Notes on the parameters

When is_llt == false:
- sigma_{drift}^2
- sigma_{cycle}^2 / sigma_{drift}^2
- AR(p) parameters

When is_llt == true:
- sigma_{drift}^2
- sigma_{trend}^2 / sigma_{drift}^2
- sigma_{cycle}^2 / sigma_{drift}^2
- AR(p) parameters

In the case in which is_rw_trend == true, sigma_{drift}^2 denotes the variance of the trend.
"""
function update_sspace_C_from_params!(params::Vector{Float64}, is_llt::Bool, sspace::KalmanSettings)    
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
    initial_univariate_decomposition(data::Union{FloatVector, JVector{Float64}}, lags::Int64, ε::Float64, is_rw_trend::Bool, is_llt::Bool; sigma_lb::Vector{Float64}=[1e-3; 1e3], sigma_ub::Vector{Float64}=[1e3; 1e4])

This function returns an initial estimate of the non-stationary and stationary components of each series. In doing so, it provides a rough starting point for the ECM algorithm.

# Note
- If both `is_rw_trend` and `is_llt` are false this function models the trend as in Kitagawa and Gersch (1996, ch. 8).
"""
function initial_univariate_decomposition(data::Union{FloatVector, JVector{Float64}}, lags::Int64, ε::Float64, is_rw_trend::Bool, is_llt::Bool; sigma_lb::Vector{Float64}=[1e-3; 1e3], sigma_ub::Vector{Float64}=[1e3; 1e4])

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

    # Initial conditions (mean)
    X0 = zeros(2+is_llt+lags);

    # Initial conditions (covariance)
    C_cycle = C[3:end, 3:end];
    DQD_cycle = Symmetric(cat(dims=[1,2], Q[2+is_llt, 2+is_llt], zeros(lags-1, lags-1)));
    P0_cycle = solve_discrete_lyapunov(C_cycle, DQD_cycle).data;
    P0_trend = Symmetric(Inf*Matrix(I, 2, 2));
    
    # Replace infs with large scalar
    P0_trend[isinf.(P0_trend)] .= 10.0^floor(Int, 2+log10(first(skipmissing(data))^2));
    
    # Merge into `P0`
    P0 = Symmetric(cat(dims=[1,2], P0_trend, P0_cycle));
    
    # Set KalmanSettings
    sspace = KalmanSettings(Array(data'), B, R, C, D, Q, X0, P0, compute_loglik=true);

    #=
    Initial values / bounds for the parameters

    When is_llt == false:
    - sigma_{drift}^2
    - sigma_{cycle}^2 / sigma_{drift}^2
    - AR(p) parameters

    When is_llt == true:
    - sigma_{drift}^2
    - sigma_{trend}^2 / sigma_{drift}^2
    - sigma_{cycle}^2 / sigma_{drift}^2
    - AR(p) parameters

    In the case in which is_rw_trend == true, sigma_{drift}^2 denotes the variance of the trend.
    =#

    params_0  = [(sigma_lb + sigma_ub)/2; 0.90; zeros(lags-1)];
    params_lb = [sigma_lb; -1*ones(lags)];
    params_ub = [sigma_ub; +1*ones(lags)];
    
    # Add `sigma_{trend}^2 / sigma_{drift}^2` entries
    if is_llt
        insert!(params_0,  2, (sigma_lb[1] + sigma_ub[1])/2);
        insert!(params_lb, 2, sigma_lb[1]);
        insert!(params_ub, 2, sigma_ub[1]);
    end
    
    # Best derivative-free option from NLopt -> NLopt.LN_SBPLX()

    # Maximum likelihood
    tuple_fmin_args = (is_llt, sspace);
    prob = OptimizationFunction(call_fmin!);
    prob = OptimizationProblem(prob, params_0, tuple_fmin_args, lb=params_lb, ub=params_ub);
    res_optim = solve(prob, NLopt.LN_SBPLX(), abstol=0.0, reltol=1e-3);
    minimizer_bounded = res_optim.u;

    #=
    # Alternative way to handle the bounds in the optimisation
    tuple_fmin_args = (params_lb, params_ub, is_llt, sspace);
    params_0_unb = [get_unbounded_logit(params_0[i], params_lb[i], params_ub[i]) for i in axes(params_0, 1)];
    prob = OptimizationFunction(call_reparametrised_fmin!);
    prob = OptimizationProblem(prob, params_0_unb, tuple_fmin_args);
    res_optim = solve(prob, NLopt.LN_SBPLX(), abstol=0.0, reltol=1e-3);
    minimizer_bounded = [get_bounded_logit(res_optim.u[i], params_lb[i], params_ub[i]) for i in axes(res_optim.u, 1)];
    =#
    
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
    initial_univariate_decomposition_kitagawa(data::Union{FloatVector, JVector{Float64}}, lags::Int64, ε::Float64, is_rw_trend::Bool; sigma_lb::Vector{Float64}=[1e-3; 1e3], sigma_ub::Vector{Float64}=[1e3; 1e4])

This function returns an initial estimate of the non-stationary and stationary components of each series. In doing so, it provides a rough starting point for the ECM algorithm.

If `is_rw_trend` is false this function models the trend as in Kitagawa and Gersch (1996, ch. 8).
"""
function initial_univariate_decomposition_kitagawa(data::Union{FloatVector, JVector{Float64}}, lags::Int64, ε::Float64, is_rw_trend::Bool; sigma_lb::Vector{Float64}=[1e-3; 1e3], sigma_ub::Vector{Float64}=[1e3; 1e4])
    trend, _, cycle, _ = initial_univariate_decomposition(data, lags, ε, is_rw_trend, false, sigma_lb=sigma_lb, sigma_ub=sigma_ub);
    return trend, cycle;
end

"""
    initial_univariate_decomposition_llt(data::Union{FloatVector, JVector{Float64}}, lags::Int64, ε::Float64, is_rw_trend::Bool; sigma_lb::Vector{Float64}=[1e-3; 1e3], sigma_ub::Vector{Float64}=[1e3; 1e4])

This function returns an initial estimate of the non-stationary and stationary components of each series. In doing so, it provides a rough starting point for the ECM algorithm.

If `is_rw_trend` is false this function models the trend as a local linear trend.
"""
function initial_univariate_decomposition_llt(data::Union{FloatVector, JVector{Float64}}, lags::Int64, ε::Float64, is_rw_trend::Bool; sigma_lb::Vector{Float64}=[1e-3; 1e3], sigma_ub::Vector{Float64}=[1e3; 1e4])
    trend, drift, cycle, _ = initial_univariate_decomposition(data, lags, ε, is_rw_trend, true, sigma_lb=sigma_lb, sigma_ub=sigma_ub);
    return trend, drift, cycle;
end

"""
    initial_detrending(Y_untrimmed::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings; use_llt::Bool=false)

Detrend each series in `Y_untrimmed` (nxT). Data can be a copy of `estim.Y`.

Return initial common trends and detrended data (after having removed initial and ending missings).
"""
function initial_detrending(Y_untrimmed::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings; use_llt::Bool=false)
    
    # Error management
    if !(isdefined(estim, :drifts_selection) &
         isdefined(estim, :ε) & 
         isdefined(estim, :lags) &
         isdefined(estim, :n_trends) &
         isdefined(estim, :trends_skeleton) &
         isdefined(estim, :verb)
        )
        
        error("This `estim` does not contain the required fields to run `initial_detrending(...)`!");
    end

    # Trim sample removing initial and ending missings (when needed)
    first_ind = findfirst(sum(ismissing.(Y_untrimmed), dims=1) .== 0)[2];
    last_ind = findlast(sum(ismissing.(Y_untrimmed), dims=1) .== 0)[2];
    Y_trimmed = Y_untrimmed[:, first_ind:last_ind] |> JMatrix{Float64};
    n_trimmed, T_trimmed = size(Y_trimmed);

    # Compute individual trends
    trends = zeros(n_trimmed, T_trimmed);
    cycles = zeros(n_trimmed, T_trimmed); 
    for i=1:n_trimmed
        verb_message(estim.verb, "Initialisation > NLopt.LN_SBPLX, variable $(i)");
        drifts_selection_ids = findall(view(estim.trends_skeleton, i, :) .!= 0.0); # (i, :) is correct since it iterates series-wise
        if length(drifts_selection_ids) > 0
            drifts_selection_id = drifts_selection_ids[findmax(i -> estim.drifts_selection[i], drifts_selection_ids)[2]];
            trends[i, :], cycles[i, :] = initial_univariate_decomposition_kitagawa(Y_trimmed[i, :], estim.lags, estim.ε, estim.drifts_selection[drifts_selection_id]==0);
        else
            cycles[i, :] .= Y_trimmed[i, :];
        end
    end

    # Compute common trends. `common_trends` is equivalent to `trends` if there aren't common trends to compute.
    common_trends = zeros(estim.n_trends, T_trimmed);

    # Looping over each trend in `common_trends`
    for i=1:estim.n_trends

        # `coordinates_current_block` denotes a vector of integers representing the series loading onto the i-th trend
        coordinates_current_block = findall(view(estim.trends_skeleton, :, i) .!= 0.0); # (:, i) is correct since it iterates trend-wise
        
        # Unadjusted estimate (dividing by the coefficients in `estim.trends_skeleton` allows to appropriately weight each series)
        common_trends[i, :] = mean(trends[coordinates_current_block, :] ./ estim.trends_skeleton[coordinates_current_block, i], dims=1);

        # Adjustment factor (consider the current trend as an increment from previous ones)
        previous_trends_mean = zeros(1, T_trimmed);

        # Loop over each series' id in `coordinates_current_block`
        for j in coordinates_current_block
            previous_trends_coordinates = findall(view(estim.trends_skeleton, j, 1:i-1) .!= 0.0); # for the j-th series, all trends before loading the i-th one
            if length(previous_trends_coordinates) > 0
                previous_trends_mean .+= sum(common_trends[previous_trends_coordinates, :], dims=1); # update `previous_trends_mean` with the total trend value for the j-th series, before loading the i-th one
            end
        end
        previous_trends_mean ./= length(coordinates_current_block); # dividing by the length of `coordinates_current_block` is correct, even when no previous trends are associated to one or more series - it can be easily proved algebrically

        # Adjusted estimate (if necessary)
        common_trends[i, :] .-= previous_trends_mean[:];
    end
    
    # Compute detrended data
    detrended_data = trends+cycles; # interpolated observables
    detrended_data .-= estim.trends_skeleton*common_trends;

    # Return output
    return common_trends, detrended_data;
end