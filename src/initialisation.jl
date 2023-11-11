"""
    update_sspace_B_from_params!(constrained_params::AbstractVector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)

Update free coordinates in `sspace.B` from `constrained_params`.
"""
function update_sspace_B_from_params!(constrained_params::AbstractVector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)
    sspace.B[coordinates_free_params_B] .= constrained_params[1:sum(coordinates_free_params_B)];
end

"""
    update_sspace_Q_from_params!(constrained_params::AbstractVector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)

Update `sspace.Q` from `constrained_params`.
"""
function update_sspace_Q_from_params!(constrained_params::AbstractVector{Float64}, coordinates_free_params_B::BitMatrix, sspace::KalmanSettings)

    # Find relevant coordinates
    n_series = size(sspace.B, 1);
    n_trends_ext = findall(sspace.B[1,:] .== 1)[2]-1;
    n_trends = n_trends_ext/2 |> Int64;

    # Trends dictionary
    trends_dict = Dict(i=>j for (j, i) in enumerate(1:2:n_trends_ext));

    # Break down parameters
    params_ratios = constrained_params[sum(coordinates_free_params_B)+1:sum(coordinates_free_params_B)+1+n_series];
    params_variances = constrained_params[sum(coordinates_free_params_B)+2+n_series:end];

    # Ratios to variances
    params_ratios_as_variances = copy(params_ratios);
    params_ratios_as_variances[end] *= params_variances[1]; # first common cycle
    for i=1:length(params_ratios_as_variances)-1
        coordinates_trends_in_B = findall(sspace.B[i, 1:n_trends_ext] .!= 0.0);   
        coordinates_trends_in_params = getindex.(Ref(trends_dict), coordinates_trends_in_B);
        params_ratios_as_variances[i] *= sum(view(params_variances, coordinates_trends_in_params)); # idio cycles
    end
    
    # Merge variances
    merged_params_variances = vcat(params_variances[1:n_trends], params_ratios_as_variances, params_variances[n_trends+1:end]);

    # Update sspace.Q
    sspace.Q.data[diagind(sspace.Q.data)] .= merged_params_variances;
end

"""
    update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0::BitMatrix, sspace::KalmanSettings)

Update `sspace.DQD` and the free entries of `sspace.P0`.
"""
function update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0::BitMatrix, sspace::KalmanSettings)

    # Update `sspace.DQD`
    sspace.DQD.data .= Symmetric(sspace.D*sspace.Q*sspace.D').data;

    # Find first cycle
    coordinates_first_cycle = findall(sspace.B[1,:] .== 1)[2];

    # C and DQD referring to cycles
    C_cycles = sspace.C[coordinates_first_cycle:end, coordinates_first_cycle:end];
    DQD_cycles = Symmetric(sspace.DQD[coordinates_first_cycle:end, coordinates_first_cycle:end]);
    coordinates_free_params_P0_cycles = @view coordinates_free_params_P0[coordinates_first_cycle:end, coordinates_first_cycle:end];

    # Update the free entries in `sspace.P0`
    sspace.P0.data[coordinates_free_params_P0] = solve_discrete_lyapunov(C_cycles, DQD_cycles).data[coordinates_free_params_P0_cycles];
end

"""
    fmin!(constrained_params::AbstractVector{Float64}, sspace::KalmanSettings, coordinates_free_params_B::BitMatrix, coordinates_free_params_P0::BitMatrix)

Return -1 times the log-likelihood function (or a large number in case of numerical problems).
"""
function fmin!(constrained_params::AbstractVector{Float64}, sspace::KalmanSettings, coordinates_free_params_B::BitMatrix, coordinates_free_params_P0::BitMatrix)

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
        is_P0_non_problematic = true;
    else
        is_P0_non_problematic = false;
    end
    
    # Regular run
    if is_Q_non_problematic && is_P0_non_problematic
        
        # Run kalman filter and return -loglik
        try            
            status = kfilter_full_sample(sspace);
            println(-status.loglik)
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
    call_fmin!(constrained_params::AbstractVector{Float64}, tuple_fmin_args::Tuple)

APIs to call `fmin!` with Tuple parameters.
"""
call_fmin!(constrained_params::AbstractVector{Float64}, tuple_fmin_args::Tuple) = fmin!(constrained_params, tuple_fmin_args...);

"""
    initial_sspace_structure(data::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings; first_step::Bool=false)

Get initial state-space parameters and relevant coordinates. 

Trends are modelled using the Kitagawa representation.
"""
function initial_sspace_structure(data::Union{FloatMatrix, JMatrix{Float64}}, estim::EstimSettings; first_step::Bool=false)

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

    if first_step
        B_common_cycles[B_common_cycles .!= 1.0] .= 0.0; # no common cycles
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
    if first_step
        C_idio_cycles[1, 1] = 0.0;                  # set to noise
        C_idio_cycles[C_idio_cycles .== 0.1] .*= 9; # set to persistent cycles (as persistent as the common cycles)
    end
    
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
    X0_trends = zeros(2*estim.n_trends);
    P0_trends = Symmetric(zeros(2*estim.n_trends, 2*estim.n_trends));

    # Loop over the trends
    for i=1:estim.n_trends
        
        # Get data in current trend
        coordinates_data_in_trend = findall(view(estim.trends_skeleton, :, i) .!= 0);
        data_in_trend = view(data, coordinates_data_in_trend, :);
        first_data_in_trend = [first(skipmissing(view(data_in_trend, j, :))) for j in axes(data_in_trend, 1)]

        # Initial conditions
        for j=1:2
            X0_trends[j+(i-1)*2] = mean(first_data_in_trend); # this allows for a weakly diffuse initialisation of the trend
            P0_trends.data[j+(i-1)*2, j+(i-1)*2] = 10.0^floor(Int, 1+log10(mean(abs.(first_data_in_trend))));
        end
    end
    
    # Setup initial conditions for the idiosyncratic cycles
    X0_idio_cycles = zeros(n_series_in_data);
    DQD_idio_cycles = Symmetric(Matrix(1.0I, n_series_in_data, n_series_in_data))
    P0_idio_cycles = solve_discrete_lyapunov(C_idio_cycles, DQD_idio_cycles).data;

    # Setup initial conditions for the common cycles
    X0_common_cycles = zeros(estim.n_cycles*estim.lags);
    DQD_common_cycles = Symmetric(D_common_cycles * Matrix(1.0I, estim.n_cycles, estim.n_cycles) * D_common_cycles');
    P0_common_cycles = solve_discrete_lyapunov(C_common_cycles, DQD_common_cycles).data;

    # Setup initial conditions
    X0 = vcat(X0_trends, X0_idio_cycles, X0_common_cycles);
    P0 = Symmetric(cat(dims=[1,2], P0_trends, P0_idio_cycles, P0_common_cycles));
    coordinates_free_params_P0 = P0 .!= 0.0;
    coordinates_free_params_P0[1:2*estim.n_trends, 1:2*estim.n_trends] .= false;

    # Return state-space matrices and relevant coordinates
    return B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0;
end

"""
    initial_detrending_step_1(Y_trimmed::JMatrix{Float64}, estim::EstimSettings, n_trimmed::Int64)

Run first step of the initialisation to find reasonable initial guesses.
"""
function initial_detrending_step_1(Y_trimmed::JMatrix{Float64}, estim::EstimSettings, n_trimmed::Int64)
    
    # Get initial state-space parameters and relevant coordinates
    B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0 = initial_sspace_structure(Y_trimmed, estim, first_step=true);
    
    # Set KalmanSettings
    sspace = KalmanSettings(Y_trimmed, B, R, C, D, Q, X0, P0, compute_loglik=true);

    # Initial guess for the parameters
    params_0 = vcat(
        1e+4*ones(1+n_trimmed),
        1e-3*ones(estim.n_trends),
    );
    params_lb = vcat(1e+2*ones(1+n_trimmed), 1e-6*ones(estim.n_trends));
    params_ub = vcat(1e+6*ones(1+n_trimmed), ones(estim.n_trends));
    
    # Maximum likelihood
    tuple_fmin_args = (sspace, coordinates_free_params_B, coordinates_free_params_P0);
    prob = OptimizationFunction(call_fmin!)
    prob = OptimizationProblem(prob, params_0, tuple_fmin_args, lb=params_lb, ub=params_ub);
    res_optim = solve(prob, NLopt.LN_SBPLX, abstol=1e-3, reltol=1e-2);

    # Update `sspace` from `res_optim`
    update_sspace_Q_from_params!(res_optim.u, coordinates_free_params_B, sspace);
    update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0, sspace);

    # Recover smoothed states
    status = kfilter_full_sample(sspace);
    smoothed_states_container, _ = ksmoother(sspace, status);
    smoothed_states = hcat(smoothed_states_container...);
    
    # Recover smoothed cycles
    smoothed_cycles = smoothed_states[2*estim.n_trends+1:2*estim.n_trends+n_trimmed, :];
    for i=1:n_trimmed
        last_state_for_ith_series = findlast(B[i, :] .== 1.0);
        if last_state_for_ith_series > 2*estim.n_trends+n_trimmed
            smoothed_cycles[i, :] .+= smoothed_states[last_state_for_ith_series, :];
        end
    end

    # Return minimizer
    return res_optim.u, smoothed_states, smoothed_cycles;
end

"""
    initialise_common_cycle(estim::EstimSettings, residual_data::FloatMatrix, coordinates_current_block::IntVector)

Initialise current common cycle via PCA.
"""
function initialise_common_cycle(estim::EstimSettings, residual_data::FloatMatrix, coordinates_current_block::IntVector)

    # Convenient shortcuts
    data_current_block = residual_data[coordinates_current_block, :];
    data_current_block_standardised = standardise(data_current_block);

    # Compute PCA loadings
    eigen_val, eigen_vect = eigen(Symmetric(cov(data_current_block_standardised, dims=2)));
    loadings = eigen_vect[:, sortperm(-abs.(eigen_val))[1]];

    # Compute PCA factor
    pc1 = permutedims(loadings)*data_current_block_standardised;

    # Rescale PCA loadings to match the original scale
    loadings .*= std(data_current_block, dims=2)[:];

    # Rescale PC1 wrt the first series
    pc1 .*= loadings[1];
    loadings ./= loadings[1];

    if estim.lags > 1

        # Backcast `pc1` first

        # Reverse `pc1` time order to predict the past (i.e., backcast)
        pc1_reversed = reverse(pc1);

        # Estimate ridge backcast coefficients
        pc1_reversed_y, pc1_reversed_x = lag(pc1_reversed, estim.lags);
        ar_coeff_backcast = pc1_reversed_y*pc1_reversed_x'/Symmetric(pc1_reversed_x*pc1_reversed_x' + estim.Γ);
        enforce_causality_and_invertibility!(ar_coeff_backcast);

        # Generate backcast for `pc1`
        for t=1:estim.lags
            backcast_x = pc1[1:estim.lags];
            pc1 = hcat(ar_coeff_backcast*backcast_x, pc1);
        end

        # Lag principal component
        pc1_y, pc1_x = lag(pc1, estim.lags);

        # Initialise complete loadings
        complete_loadings = zeros(length(loadings), estim.lags);

        # Regress one variable at the time on `pc1`
        pc1_x_shifted_with_backcast = vcat(pc1_y, pc1_x[1:end-1, :]);
        pc1_x_shifted = pc1_x_shifted_with_backcast[:, estim.lags+1:end];
        for i in axes(data_current_block, 1)
            if i == 1
                complete_loadings[1, 1] = 1.0; # identification
            else
                data_current_block_yi, _ = lag(permutedims(data_current_block[i, :]), estim.lags);
                complete_loadings[i, :] = data_current_block_yi*pc1_x_shifted'/Symmetric(pc1_x_shifted*pc1_x_shifted' + estim.Γ);
            end
        end

        # Explained data
        explained_data = complete_loadings*pc1_x_shifted_with_backcast;

    else
        complete_loadings = loadings;
        explained_data = complete_loadings*pc1;
    end

    # Return output
    return complete_loadings, explained_data;
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
    n_trimmed = size(Y_trimmed, 1);

    # Recover initial guess from step 1
    println("Initialisation > running step 1")
    params_0, _, smoothed_cycles_0 = MessyTimeSeriesOptim.initial_detrending_step_1(Y_trimmed, estim, n_trimmed);

    # Get initial state-space parameters and relevant coordinates
    B, R, C, D, Q, X0, P0, coordinates_free_params_B, coordinates_free_params_P0 = MessyTimeSeriesOptim.initial_sspace_structure(Y_trimmed, estim);

    # Determine which series can load on each cycle
    boolean_coordinates_blocks = (estim.cycles_skeleton .!= 0) .| estim.cycles_free_params;
    coordinates_blocks = [findall(boolean_coordinates_blocks[:, i]) for i=1:estim.n_cycles];

    # Initialise common cycles iteratively via PCA
    residual_data = copy(smoothed_cycles_0);

    # Loop over common cycles
    for i=1:estim.n_cycles

        # Pointers
        coordinates_current_block = coordinates_blocks[i];

        # Loadings for the i-th common cycle
        complete_loadings, explained_data = MessyTimeSeriesOptim.initialise_common_cycle(estim, residual_data, coordinates_current_block);
        
        # Position of the i-th common cycle in the state-space representation
        coordinates_pc1 = 1 + (i-1)*estim.lags + 2*estim.n_trends + n_trimmed;

        # Position `complete_loadings` in `B`
        B[:, coordinates_pc1:coordinates_pc1+estim.lags-1] = complete_loadings;

        # Recompute residual data
        residual_data[coordinates_current_block, :] .-= explained_data;
    end
    
    # Set KalmanSettings
    sspace = KalmanSettings(Y_trimmed, B, R, C, D, Q, X0, P0, compute_loglik=true);

    # `B` lower and upper bounds
    vec_B = B[coordinates_free_params_B];
    vec_B_lb = [ifelse(vec_B[i] > 0, vec_B[i]/10, vec_B[i]*10) for i in axes(vec_B, 1)];
    vec_B_ub = [ifelse(vec_B[i] > 0, vec_B[i]*10, vec_B[i]/10) for i in axes(vec_B, 1)];

    # Update `params_0`
    params_0 = vcat(vec_B, params_0);
    params_lb = vcat(vec_B_lb, 1e+2*ones(1+n_trimmed), 1e-6*ones(estim.n_trends));
    params_ub = vcat(vec_B_ub, 1e+6*ones(1+n_trimmed), ones(estim.n_trends));
    
    # Maximum likelihood
    println("Initialisation > running step 2")
    tuple_fmin_args = (sspace, coordinates_free_params_B, coordinates_free_params_P0);
    prob = OptimizationFunction(call_fmin!)
    prob = OptimizationProblem(prob, params_0, tuple_fmin_args, lb=params_lb, ub=params_ub);
    res_optim = solve(prob, NLopt.LN_SBPLX, abstol=1e-3, reltol=1e-2);

    # Update sspace accordingly
    update_sspace_B_from_params!(res_optim.u, coordinates_free_params_B, sspace);
    update_sspace_Q_from_params!(res_optim.u, coordinates_free_params_B, sspace);
    update_sspace_DQD_and_P0_from_params!(coordinates_free_params_P0, sspace);

    # Recover smoothed states
    status = kfilter_full_sample(sspace);
    smoothed_states_container, _ = ksmoother(sspace, status);
    smoothed_states = hcat(smoothed_states_container...);

    # Return output
    return smoothed_states, sspace;
end
