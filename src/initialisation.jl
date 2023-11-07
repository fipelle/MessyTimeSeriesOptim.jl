"""
    update_sspace_B_from_params!(params::Vector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)

Update free coordinates in `sspace.B` from `params`.
"""
function update_sspace_B_from_params!(params::Vector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)
    sspace.B[coordinates_free_params_B] .= params[1:sum(coordinates_free_params_B)];
end

"""
    update_sspace_Q_from_params!(params::Vector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)

Update `sspace.Q` from `params`.
"""
function update_sspace_Q_from_params!(params::Vector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)
    sspace.Q.data[diagind(sspace.Q.data)] .= params[sum(coordinates_free_params_B)+1:end];
end

"""
    update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0::BitMatrix, sspace::KalmanSettings)

Update `sspace.DQD` and the free entries of `sspace.P0` from `params`.
"""
function update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0::BitMatrix, sspace::KalmanSettings)

    # Update `sspace.DQD`
    sspace.DQD.data .= Symmetric(sspace.D*sspace.Q*sspace.D').data;

    # Find first cycle
    coordinates_first_cycle = findfirst(coordinates_free_params_P0);
    println(coordinates_first_cycle);
    println(findall(sspace.B[1,:] .== 1)[2])
    error("AAA");

    # Idio cycles
    
    # Common cycles

    # Update the free entries in `sspace.P0`
    sspace.P0.data[coordinates_free_params_P0] = solve_discrete_lyapunov(sspace.C, sspace.DQD).data[coordinates_free_params_P0];
end

"""
    fmin!(constrained_params::Vector{Float64}, sspace::KalmanSettings, coordinates_free_params_B::BitMatrix, coordinates_free_params_P0::BitMatrix)

Return -1 times the log-likelihood function (or a large number in case of numerical problems).
"""
function fmin!(constrained_params::Vector{Float64}, sspace::KalmanSettings, coordinates_free_params_B::BitMatrix, coordinates_free_params_P0::BitMatrix)

    # Update sspace accordingly
    update_sspace_B_from_params!(constrained_params, coordinates_free_params_B, sspace);
    update_sspace_Q_from_params!(constrained_params, coordinates_free_params_B, sspace);

    # Determine whether Q is problematic
    if (sum(isnan.(sspace.Q)) == 0) && (sum(isinf.(sspace.Q)) == 0)
        is_Q_non_problematic = true;
        update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0, sspace);
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
    if is_Q_non_problematic && is_P0_non_problematic
        
        # Run kalman filter and return -loglik
        try            
            status = kfilter_full_sample(sspace);
            return -status.loglik;
        
        # Problematic run
        catch kf_run_error
            if isa(kf_run_error, DomainError)
                return 1/eps();
            else
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
    initial_sspace_structure(data::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings)

Get initial state-space parameters and relevant coordinates. 

Trends are modelled using the Kitagawa representation.
"""
function initial_sspace_structure(data::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings)

    # `n` for initialisation (it may differ from the one in `estim`)
    n_series_in_data = size(data, 1);

    # Setup loading structure for the trends (common and idiosyncratic)
    B_trends = kron(estim.trends_skeleton, [1.0 0.0]);

    # Setup loadings structure for the idiosyncratic cycles
    B_idio_cycles = Matrix(1.0I, n_series_in_data, n_series_in_data);

    # Setup loading structure for the cycles, employing the relevant identification scheme
    B_common_cycles = kron(estim.cycles_skeleton, ones(1, estim.lags));
    B_common_cycles[1, :] .= 0.0;
    for i in [1+(i-1)*estim.lags for i in 1:estim.n_cycles]
        B_common_cycles[1, i] = 1.0;
    end

    # Setup loading matrix
    B = hcat(B_trends, B_idio_cycles, B_common_cycles);
    coordinates_free_params_B = B .> 1.0;

    # Setup covariance matrix measurement error
    R = 1e-4*I;

    # Setup transition matrix for the trends (common and idiosyncratic)
    C_trends_template = [2.0 -1.0; 1.0 0.0];
    C_trends = cat(dims=[1,2], [C_trends_template for i in 1:estim.n_trends]...);

    # Setup transition matrix for the idiosyncratic cycles
    C_idio_cycles = Matrix(0.1I, n_series_in_data, n_series_in_data);

    # Setup transition matrix for the common cycles
    C_common_cycles_template = companion_form([0.9 zeros(1, estim.lags-1)], extended=false);
    C_common_cycles = cat(dims=[1,2], [C_common_cycles_template for i in 1:estim.n_cycles]...);

    # Setup transition matrix
    C = cat(dims=[1,2], C_trends, C_idio_cycles, C_common_cycles);

    # Setup covariance matrix of the states' innovation
    # NOTE: all diagonal elements are estimated during the initialisation
    Q = Symmetric(Matrix(1.0I, estim.n_trends + n_series_in_data + estim.n_cycles, estim.n_trends + n_series_in_data + estim.n_cycles));

    # Setup selection matrix D for the trends
    D_trends_template = [1.0; 0.0];
    D_trends = cat(dims=[1,2], [D_trends_template for i in 1:estim.n_trends]...);

    # Setup selection matrix D for idiosyncratic cycles
    D_idio_cycles = Matrix(1.0I, n_series_in_data, n_series_in_data);

    # Setup selection matrix D for the common cycles
    D_common_cycles_template = zeros(estim.lags);
    D_common_cycles_template[1] = 1.0;
    D_common_cycles = cat(dims=[1,2], [D_common_cycles_template for i in 1:estim.n_cycles]...);

    # Setup selection matrix D
    D = cat(dims=[1,2], D_trends, D_idio_cycles, D_common_cycles);

    # Setup initial conditions for the trends
    X0_trends = zeros(estim.n_trends);
    P0_trends = Symmetric(zeros(estim.n_trends, estim.n_trends));

    # Loop over the trends
    for i=1:estim.n_trends
        
        # Get data in current trend
        coordinates_data_in_trend = findall(view(estim.trends_skeleton, :, i) .!= 0);
        data_in_trend = view(data, coordinates_data_in_trend, :);
        first_data_in_trend = [first(skipmissing(view(data_in_trend, j, :))) for j in axes(data_in_trend, 1)]

        # Initial conditions
        X0_trends[i] = mean(first_data_in_trend); # this allows for a weakly diffuse initialisation of the trend
        P0_trends.data[i, i] = 10.0^floor(Int, 1+log10(mean(abs.(first_data_in_trend))));
    end

    # Setup initial conditions for the idiosyncratic cycles
    X0_idio_cycles = zeros(n_series_in_data);
    DQD_idio_cycles = Symmetric(Matrix(1.0I, n_series_in_data, n_series_in_data))
    P0_idio_cycles = solve_discrete_lyapunov(C_idio_cycles, DQD_idio_cycles).data;

    # Setup initial conditions for the common cycles
    X0_common_cycles = zeros(estim.n_cycles);
    DQD_common_cycles = Symmetric(D_common_cycles * Matrix(1.0I, estim.n_cycles, estim.n_cycles) * D_common_cycles');
    P0_common_cycles = solve_discrete_lyapunov(C_common_cycles, DQD_common_cycles).data;

    # Setup initial conditions
    X0 = vcat(X0_trends, X0_idio_cycles, X0_common_cycles);
    P0 = Symmetric(cat(dims=[1,2], P0_trends, P0_idio_cycles, P0_common_cycles));
    coordinates_free_params_P0 = P0 .!= 0.0;
    coordinates_free_params_P0[1:estim.n_trends, 1:estim.n_trends] .= false;

    # Return state-space matrices and relevant coordinates
    return B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0;
end

"""
    initial_detrending(Y_untrimmed::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings)

Detrend each series in `Y_untrimmed` (nxT). Data can be a copy of `estim.Y`.

Return initial common trends and detrended data (after having removed initial and ending missings).
"""
function initial_detrending(Y_untrimmed::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings)
    
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

    # Get initial state-space parameters and relevant coordinates
    B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0 = initial_sspace_structure(Y_trimmed, estim);

    # Set KalmanSettings
    sspace = KalmanSettings(Y_trimmed, B, R, C, D, Q, X0, P0, compute_loglik=true);

    # Initial guess for the parameters
    params_0  = ones(sum(coordinates_free_params_B)+size(Q, 1));
    params_lb = params_0/1000
    params_ub = params_0*1000
    
    # Best derivative-free option from NLopt -> NLopt.LN_SBPLX()

    # Maximum likelihood
    tuple_fmin_args = (sspace, coordinates_free_params_B, coordinates_free_params_P0);
    prob = OptimizationFunction(call_fmin!);
    prob = OptimizationProblem(prob, params_0, tuple_fmin_args, lb=params_lb, ub=params_ub);
    res_optim = solve(prob, NLopt.LN_SBPLX(), abstol=0.0, reltol=1e-3);
    minimizer_bounded = res_optim.u;

    # Update sspace accordingly
    update_sspace_B_from_params!(minimizer_bounded, coordinates_free_params_B, sspace);
    update_sspace_Q_from_params!(minimizer_bounded, coordinates_free_params_B, sspace);
    update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0, sspace);
    
    # Retrieve optimal states
    status = kfilter_full_sample(sspace);
    smoothed_states, _ = ksmoother(sspace, status);
    smoothed_states_matrix = hcat(smoothed_states...);
        
    # Return output
    return smoothed_states_matrix;
end
