#=
--------------------------------------------------------------------------------------------------------------------------------
Types and constructors
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    DFMSettings(...)

Define an immutable structure used to initialise the estimation routine for a DFM.

# Arguments
- `Y`: observed measurements (`nxT`)
- `n`: Number of series
- `T`: Number of observations
- `lags`: Order of the autoregressive polynomial of each common cycle
- `n_trends`: Number of trends
- `n_drifts`: Number of drifts
- `n_cycles`: Number of cycles
- `n_non_stationary`: n_trends + n_drifts
- `m`: n_non_stationary + n_cycles*lags + n
- `trends_skeleton`: The basic structure for the trends loadings (or nothing)
- `cycles_skeleton`: The basic structure for the cycles loadings (or nothing)
- `drifts_selection`: BitArray{1} identifying which trend has a drift to estimate (or nothing)
- `trends_free_params`: BitArray{2} identifying the trend loadings to estimate (or nothing)
- `cycles_free_params`: BitArray{2} identifying the cycle loadings to estimate (or nothing)
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `Γ`: Diagonal matrix used to input the hyperparameters in the penalty computation for the common cycles
- `Γ_idio`: Diagonal matrix used to input the hyperparameters in the penalty computation for the idiosyncratic cycles
- `Γ_extended`: Diagonal matrix used to input the hyperparameters in the estimation
- `ε`: Small number (default: 1e-4)
- `tol`: tolerance used to check convergence (default: 1e-4)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual estimation algorithm (default: 2)
- `check_quantile`: check the quantile of the relative change of the parameters for convergence purposes (default: false)
- `verb`: Verbose output (default: true)
"""
struct DFMSettings <: EstimSettings
    Y::Union{FloatMatrix, JMatrix{Float64}}
    n::Int64
    T::Int64
    lags::Int64
    n_trends::Int64
    n_drifts::Int64
    n_cycles::Int64
    n_non_stationary::Int64
    m::Int64
    trends_skeleton::Union{FloatMatrix, Nothing}
    cycles_skeleton::Union{FloatMatrix, Nothing}
    drifts_selection::Union{BitVector, Nothing}
    trends_free_params::Union{BitMatrix, Nothing}
    cycles_free_params::Union{BitMatrix, Nothing}
    λ::Float64
    α::Float64
    β::Float64
    Γ::DiagMatrix
    Γ_idio::Float64
    Γ_extended::DiagMatrix
    ε::Float64
    tol::Float64
    max_iter::Int64
    prerun::Int64
    check_quantile::Bool
    verb::Bool
end

#=
DFMSettings constructor (no input provided for basic skeletons and free parameters)
- This constructor returns the default implementation for a stationary DFM
=#
function DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, lags::Int64, n_cycles::Int64, λ::Float64, α::Float64, β::Float64; ε::Float64=1e-4, tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, check_quantile::Bool=false, verb::Bool=true)

    # Make sure there is at least one common cycle to estimate
    check_bounds(n_cycles, 0);

    # Compute missing dimensions
    n, T = size(Y);
    n_trends = 0;
    n_drifts = 0;
    n_non_stationary = 0;
    m = n_non_stationary + n_cycles*lags + n;

    # Compute missing input for estimation
    cycles_skeleton = hcat([[zeros(i-1); 1; 2*ones(n-i)] for i=1:n_cycles]...);
    cycles_free_params = cycles_skeleton .> 1;

    # Compute penalty data
    Γ = build_Γ(1, lags, λ, β);
    Γ_idio = build_Γ(1, 1, λ, β)[1];
    Γ_extended = cat(dims=[1,2], [Γ_idio for i=1:n]..., [Γ for i=1:n_cycles]...) |> Array |> Diagonal;

    # Return DFMSettings
    return DFMSettings(Y, n, T, lags, n_trends, n_drifts, n_cycles, n_non_stationary, m, nothing, cycles_skeleton, nothing, nothing, cycles_free_params, λ, α, β, Γ, Γ_idio, Γ_extended, ε, tol, max_iter, prerun, check_quantile, verb);
end

#=
DFMSettings constructor (input provided only for the cycles skeleton and free parameters)
- This constructor returns the default custom implementation for a stationary DFM
=#
function DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, lags::Int64, cycles_skeleton::FloatMatrix, cycles_free_params::BitMatrix, λ::Float64, α::Float64, β::Float64; ε::Float64=1e-4, tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, check_quantile::Bool=false, verb::Bool=true)

    # Compute missing dimensions
    n, T = size(Y);
    n_trends = 0;
    n_drifts = 0;
    n_cycles = size(cycles_skeleton, 2);
    n_non_stationary = 0;
    m = n_non_stationary + n_cycles*lags + n;

    # Compute penalty data
    Γ = build_Γ(1, lags, λ, β);
    Γ_idio = build_Γ(1, 1, λ, β)[1];
    Γ_extended = cat(dims=[1,2], [Γ_idio for i=1:n]..., [Γ for i=1:n_cycles]...) |> Array |> Diagonal;

    # Return DFMSettings
    return DFMSettings(Y, n, T, lags, n_trends, n_drifts, n_cycles, n_non_stationary, m, nothing, cycles_skeleton, nothing, nothing, cycles_free_params, λ, α, β, Γ, Γ_idio, Γ_extended, ε, tol, max_iter, prerun, check_quantile, verb);
end

#=
DFMSettings constructor (input provided for the cycles and trends skeleton / free parameters)
- This constructor returns the default custom implementation for a non-stationary DFM
=#
function DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, lags::Int64, trends_skeleton::FloatMatrix, cycles_skeleton::FloatMatrix, drifts_selection::BitVector, trends_free_params::BitMatrix, cycles_free_params::BitMatrix, λ::Float64, α::Float64, β::Float64; ε::Float64=1e-4, tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, check_quantile::Bool=false, verb::Bool=true)

    # Compute missing dimensions
    n, T = size(Y);
    n_trends = size(trends_skeleton, 2);
    n_drifts = sum(drifts_selection);
    n_cycles = size(cycles_skeleton, 2);
    n_non_stationary = n_trends + n_drifts;
    m = n_non_stationary + n_cycles*lags + n;

    # Compute penalty data
    Γ = build_Γ(1, lags, λ, β);
    Γ_idio = build_Γ(1, 1, λ, β)[1];
    Γ_extended = cat(dims=[1,2], Diagonal(zeros(n_non_stationary, n_non_stationary)), [Γ_idio for i=1:n]..., [Γ for i=1:n_cycles]...) |> Array |> Diagonal;

    # Return DFMSettings
    return DFMSettings(Y, n, T, lags, n_trends, n_drifts, n_cycles, n_non_stationary, m, trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params, λ, α, β, Γ, Γ_idio, Γ_extended, ε, tol, max_iter, prerun, check_quantile, verb);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Initialisation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    check_model_bounds(estim::DFMSettings)

Check whether the arguments for the estimation of a DFM model are correct.
"""
function check_model_bounds(estim::DFMSettings)

    # Simple bounds
    check_bounds(estim.lags, 0);
    check_bounds(estim.n_cycles, 0);
    check_bounds(estim.λ, 0);
    check_bounds(estim.α, 0, 1);
    check_bounds(estim.β, 1);
    check_bounds(estim.max_iter, 3);
    check_bounds(estim.max_iter, estim.prerun);
    check_bounds(estim.n, 2);
    
    # Advanced checks

    if eltype(estim.trends_skeleton) <: FloatMatrix
        if size(estim.trends_skeleton, 2) != size(estim.trends_free_params, 2)
            throw(DomainError);
        end

        if sum(estim.trends_free_params .!= 0.0) != 0
            error("The estimation of the trends' measurement coefficients is not supported yet!"); # TBD: it may be easier to implement by using the direct Kitagawa representation with 3 states rather than 4
        end

        for row in eachrow(estim.trends_skeleton)
            if length(findall(row .!= 0.0)) > 1
                error("Linear combinations of different trends are not supported yet!"); # TBD: implement it!
            end
        end
    end

    if eltype(estim.cycles_skeleton) <: FloatMatrix
        if size(estim.cycles_skeleton, 2) != size(estim.cycles_free_params, 2)
            throw(DomainError);
        end
    end

    if eltype(estim.drifts_selection) <: BitVector
        if size(estim.trends_skeleton, 2) != length(estim.drifts_selection)
            throw(DomainError);
        end
    end

    if maximum(diff(sum(estim.cycles_skeleton .!= 0, dims=2), dims=1)) .> 1
        throw(error("The structure of `cycles_skeleton` is incorrect. Cycles should be added incrementally."));
    end

    if maximum(diff(sum(estim.cycles_free_params .!= 0, dims=2), dims=1)) .> 1
        throw(error("The structure of `cycles_free_params` breaks `cycles_skeleton`."));
    end
end

"""
    initialise_trends(estim::DFMSettings)

Return initialised parameters for the trends.
"""
function initialise_trends(estim::DFMSettings, common_trends::Union{FloatMatrix, JMatrix{Float64}})

    # TBD: Check with one trend only!!!!!
    
    # Guesses for the variances
    variance_rw_trends = std_skipmissing(diff(common_trends, dims=2))[:].^2;
    variance_i2_trends = std_skipmissing(@views common_trends[:, 3:end]-2*common_trends[:, 2:end-1]+common_trends[:, 1:end-2])[:].^2;
    
    #=
    Memory pre-allocation for the state-space parameters linked to the non-stationary components
    Note: similarly to Watson and Engle (1983), estim.n_non_stationary lags of the main non-stationary components are added to compute PPs in the ECM algorithm
    =#

    B_trends = zeros(estim.n, 2*estim.n_non_stationary);
    C_trends = zeros(2*estim.n_non_stationary, 2*estim.n_non_stationary);
    D_trends = zeros(2*estim.n_non_stationary, estim.n_trends);
    Q_trends = zeros(estim.n_trends, estim.n_trends);
    X0_trends = zeros(2*estim.n_non_stationary);
    P0_trends = Matrix(Inf*I, 2*estim.n_non_stationary, 2*estim.n_non_stationary); # initialisation for P0_trends finalised in initialise(...) to form a stabler P0

    # Initialise counter
    i = 0;

    # Loop over the trends
    for j=1:estim.n_trends

        B_trends[:, 1+i] = estim.trends_skeleton[:,j];

        # Smooth I(2) trend
        if estim.drifts_selection[j]
            C_trends[1+i:4+i, 1+i:4+i] = [1 0 1 0; 1 0 0 0; 0 0 1 0; 0 0 1 0];
            D_trends[3+i, j] = 1;
            Q_trends[j, j] = variance_i2_trends[j];
            i += 4;
        
        # Driftless random walk trend
        else
            C_trends[1+i:2+i, 1+i:2+i] = [1 0; 1 0];
            D_trends[1+i, j] = 1;
            Q_trends[j, j] = variance_rw_trends[j];
            i += 2;
        end
    end

    # Return output
    return B_trends, C_trends, D_trends, Q_trends, X0_trends, P0_trends;
end

"""
    initialise_common_cycle(estim::DFMSettings, residual_data::FloatMatrix, coordinates_current_block::IntVector)

Initialise current common cycle via PCA.
"""
function initialise_common_cycle(estim::DFMSettings, residual_data::FloatMatrix, coordinates_current_block::IntVector)

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
    loadings /= loadings[1];

    # Estimate ridge autoregressive coefficients
    pc1_y, pc1_x = lag(pc1, estim.lags);
    ar_coeff = pc1_y*pc1_x'/Symmetric(pc1_x*pc1_x' + estim.Γ);
    enforce_causality_and_invertibility!(ar_coeff);

    # Estimate var-cov matrix of the residuals
    ar_residuals = pc1_y - ar_coeff*pc1_x;
    ar_residuals_variance = (ar_residuals*ar_residuals')/length(ar_residuals);

    if estim.lags > 1

        #=
        Data to initialise dynamic loadings
        =#
        
        # Regular VAR configuration
        B = [1.0 zeros(1, estim.lags-1)];
        R = Symmetric(estim.ε * ones(1,1));
        C = companion_form(ar_coeff, extended=false);
        D = [1.0; zeros(estim.lags-1, 1)];
        Q = Symmetric(ar_residuals_variance);

        # Add estim.lags missings before pc1
        sspace = KalmanSettings(hcat(missing .* ones(1, estim.lags-1), pc1), B, R, C, D, Q, compute_loglik=false);
        status = kfilter_full_sample(sspace);
        smoothed_states, _ = ksmoother(sspace, status);

        # Backcast factor up to t=-estim.lags
        pc1_extended = mapreduce(Xt -> sspace.B*Xt, hcat, smoothed_states);
        pc1_extended_x = vcat([pc1_extended[:, estim.lags-j:end-j] for j=0:estim.lags-1]...);
        pc1_extended_x_current = permutedims(pc1_extended_x[1, :]);
        pc1_extended_x_lags = pc1_extended_x[2:end, :];
        if estim.lags == 2
            pc1_extended_x_lags = permutedims(pc1_extended_x_lags);
        end

        #=
        Initialise dynamic loadings
        =#

        # Update `data_current_block` to account for the part explained by `pc1`
        data_current_block -= loadings*pc1_extended_x_current;

        # Compute complete loadings
        complete_loadings = zeros(length(loadings), estim.lags);
        complete_loadings[:, 1] = loadings;
        complete_loadings[2:end, 2:end] = data_current_block[2:end, :]*pc1_extended_x_lags'/Symmetric(pc1_extended_x_lags*pc1_extended_x_lags');
        # TBD: Improve previous line to include a ridge penalty, if possible

        # Explained data
        explained_data = complete_loadings*pc1_extended_x;

    else
        complete_loadings = loadings;
        explained_data = complete_loadings*pc1;
    end

    # Return output
    return complete_loadings, explained_data, ar_coeff, ar_residuals_variance;
end

"""
    initialise_cycles(estim::DFMSettings, data::FloatMatrix)

Return initialised parameters for the cycles.
"""
function initialise_cycles(estim::DFMSettings, data::FloatMatrix)

    #=
    Memory pre-allocation for the state-space parameters referring to the cycles (both common and idiosyncratic)
    Note: similarly to Watson and Engle (1983), n_cycles+n lags of the main stationary components are added to compute PPs in the ECM algorithm
    =#

    B_cycles = zeros(estim.n, estim.n_cycles*estim.lags + estim.n_cycles + 2*estim.n);
    C_cycles = zeros(estim.n_cycles*estim.lags + estim.n_cycles + 2*estim.n, estim.n_cycles*estim.lags + estim.n_cycles + 2*estim.n);
    D_cycles = zeros(estim.n_cycles*estim.lags + estim.n_cycles + 2*estim.n, estim.n_cycles + estim.n);
    Q_cycles = zeros(estim.n_cycles+estim.n, estim.n_cycles+estim.n);
    X0_cycles = zeros(size(B_cycles,2));
    P0_cycles = zeros(estim.n_cycles*estim.lags + estim.n_cycles + 2*estim.n, estim.n_cycles*estim.lags + estim.n_cycles + 2*estim.n);

    # Set-up loadings for idiosyncratic components
    B_cycles[:, 1:2*estim.n] = cat(dims=[1,2], [[1 0] for i=1:estim.n]...);

    # Determine which variables can load on each cycle
    boolean_coordinates_blocks = (estim.cycles_skeleton .!= 0) .| estim.cycles_free_params;
    coordinates_blocks = [findall(boolean_coordinates_blocks[:,i]) for i=1:estim.n_cycles];

    # Initialise common cycles iteratively via PCA
    residual_data = copy(data);

    for i=1:estim.n_cycles

        # Pointers
        coordinates_current_block = coordinates_blocks[i];
        
        # Initialise i-th common cycle
        complete_loadings, explained_data, ar_coeff, ar_residuals_variance = initialise_common_cycle(estim, residual_data, coordinates_current_block);

        #=
        Update state-space representation
        Note: the idiosyncratic components are placed before the common ones, as in the reference paper
        =#
        
        # Position of the i-th common cycle in the state-space representation
        coordinates_pc1 = 1+(i-1)*(estim.lags+1)+2*estim.n;

        # Convenient short cut for `C_cycles` and `P0_cycles`
        current_companion = companion_form(ar_coeff, extended=true);
        current_cov_companion = Symmetric(cat(dims=[1,2], ar_residuals_variance[1], zeros(estim.lags, estim.lags)));

        # Update coefficients
        B_cycles[coordinates_current_block, coordinates_pc1:coordinates_pc1+estim.lags-1] = complete_loadings;
        C_cycles[coordinates_pc1:coordinates_pc1+estim.lags, coordinates_pc1:coordinates_pc1+estim.lags] = current_companion;
        D_cycles[coordinates_pc1, i+estim.n] = 1; 
        Q_cycles[i+estim.n, i+estim.n] = ar_residuals_variance[1];

        # Update `P0_cycles` in a block-wise fashion
        P0_cycles[coordinates_pc1:coordinates_pc1+estim.lags, coordinates_pc1:coordinates_pc1+estim.lags] = Array(solve_discrete_lyapunov(current_companion, current_cov_companion));

        # Recompute residual data
        residual_data[coordinates_current_block, :] .-= explained_data;
    end

    # Initialise idiosyncratic cycles as white noises
    for i=1:estim.n
        
        # Estimate ridge autoregressive coefficients
        idio_y, idio_x = lag(permutedims(residual_data[i, :]), 1);
        #=
        # Remove comment to initialise as AR(1)
        idio_coeff = idio_y*idio_x'/Symmetric(idio_x*idio_x' .+ estim.Γ[1,1]);
        MessyTimeSeriesOptim.enforce_causality_and_invertibility!(idio_coeff);
        =#
        idio_coeff = zeros(1,1);

        # Estimate var-cov matrix of the residuals
        idio_resid = idio_y - idio_coeff*idio_x;
        idio_resid_var = (idio_resid*idio_resid')/length(idio_resid);

        #=
        Update state-space representation
        Note: the idiosyncratic components are placed before the common ones, as in the reference paper
        =#

        # Position of the i-th idiosyncratic component
        coordinates_idio = 1+(i-1)*2; # place the idiosyncratic components before the common ones as in the reference paper

        # Convenient short cut for `C_cycles` and `P0_cycles`
        current_companion = companion_form(idio_coeff, extended=true);
        current_cov_companion = Symmetric(cat(dims=[1,2], idio_resid_var[1], zeros(1,1)));

        # Update coefficients
        C_cycles[coordinates_idio:coordinates_idio+1, coordinates_idio:coordinates_idio+1] = current_companion;
        D_cycles[coordinates_idio, i] = 1; 
        Q_cycles[i, i] = idio_resid_var[1];

        # Update `P0_cycles` in a block-wise fashion
        P0_cycles[coordinates_idio:coordinates_idio+1, coordinates_idio:coordinates_idio+1] = Array(solve_discrete_lyapunov(current_companion, current_cov_companion));
    end

    # Return output
    return B_cycles, C_cycles, D_cycles, Q_cycles, X0_cycles, P0_cycles;
end

"""
    initialise(estim::DFMSettings, trends_skeleton::Nothing)

Initialise the ECM algorithm for stationary DFM models.

    initialise(estim::DFMSettings, trends_skeleton::FloatMatrix)

Initialise the ECM algorithm for non-stationary DFM models.

    initialise(estim::DFMSettings)

Initialise the ECM algorithm for DFM models.

# Arguments
- `estim`: settings used for the estimation

# References
Pellegrino (2022)
"""
function initialise(estim::DFMSettings, trends_skeleton::Nothing)

    # Interpolate data
    data = interpolate_series(estim.Y, estim.n, estim.T);

    # Build state-space parameters
    B, C, D, Q, X0, P0 = initialise_cycles(estim, data);
    Q = Symmetric(Q);
    R = Symmetric(Matrix(estim.ε * I, estim.n, estim.n));

    # Generate sspace
    sspace = KalmanSettings(estim.Y, B, R, C, D, Q, X0, P0, compute_loglik=false);

    # TBA coordinates etc
    error("This is not finalised yet!");

    # Return output
    return sspace;
end

function initialise(estim::DFMSettings, trends_skeleton::FloatMatrix)

    # Trim sample removing initial and ending missings (when needed)
    first_ind = findfirst(sum(ismissing.(estim.Y), dims=1) .== 0)[2];
    last_ind = findlast(sum(ismissing.(estim.Y), dims=1) .== 0)[2];
    Y_trimmed = estim.Y[:, first_ind:last_ind] |> JMatrix{Float64};
    T_trimmed = size(Y_trimmed, 2);

    # Compute individual trends
    trends = zeros(estim.n, T_trimmed);
    cycles = zeros(estim.n, T_trimmed);
    for i=1:estim.n
        @info("Initialisation > Newton's method, variable $(i)");
        drifts_selection_id = findfirst(view(estim.trends_skeleton, i, :) .!= 0.0); # (i, :) is correct since it iterates series-wise
        trends[i, :], cycles[i, :] = initial_univariate_decomposition_kitagawa(Y_trimmed[i, :], estim.lags, estim.ε, estim.drifts_selection[drifts_selection_id]==0);
    end

    # Compute common trends. `common_trends` is equivalent to `trends` if there aren't common trends to compute.
    common_trends = ones(estim.n_trends, T_trimmed);
    for i=1:estim.n_trends
        coordinates_current_block = findall(view(estim.trends_skeleton, :, i) .!= 0.0); # (:, i) is correct since it iterates trend-wise
        common_trends[i, :] = mean(trends[coordinates_current_block, :] ./ estim.trends_skeleton[coordinates_current_block, i], dims=1);
    end

    # Compute detrended data
    detrended_data = trends+cycles; # interpolated observables
    detrended_data .-= estim.trends_skeleton*common_trends;
    
    # Build state-space parameters
    B_trends, C_trends, D_trends, Q_trends, X0_trends, P0_trends = initialise_trends(estim, common_trends);
    B_cycles, C_cycles, D_cycles, Q_cycles, X0_cycles, P0_cycles = initialise_cycles(estim, detrended_data);
    B = hcat(B_trends, B_cycles);
    R = estim.ε * I;
    C = cat(dims=[1,2], C_trends, C_cycles);
    D = cat(dims=[1,2], D_trends, D_cycles);
    Q = Symmetric(cat(dims=[1,2], Q_trends, Q_cycles));
    
    # Finalise initialisation of P0_trends
    oom_maximum_P0_cycles = floor(Int, log10(maximum(P0_cycles)));  # order of magnitude of the largest entry in P0_cycles
    P0_trends[isinf.(P0_trends)] .= 10.0^(oom_maximum_P0_cycles+4); # non-stationary components (variances)
    # TBD: for now, P0_trends is a bit higher than expected (1e8 with the current example) - check the data standardisation

    # Initial conditions
    X0 = vcat(X0_trends, X0_cycles);
    P0 = Symmetric(cat(dims=[1,2], P0_trends, P0_cycles));

    #@infiltrate # check just the cat order for B, R, C, D, Q, X0 and P0 - the internal structure is fine. Next, go to the next breakpoint.

    # Generate sspace
    sspace = KalmanSettings(estim.Y, B, R, C, D, Q, X0, P0, compute_loglik=false);

    # `coordinates_transition_current` identifies the states for which there is an associated variance that is allowed to differ from zero.
    coordinates_transition_current = findall(sum(D, dims=2)[:] .== 1); # clearly, this and the following coordinates do not have the ids for the level of the trends, since dfm.jl uses Kitagawa's smooth parametrisation

    # Coordinates per type (i.e., `coordinates_transition_current` breakdown)
    coordinates_transition_non_stationary = coordinates_transition_current[coordinates_transition_current .<= 2*estim.n_non_stationary];
    coordinates_transition_stationary = coordinates_transition_current[coordinates_transition_current .> 2*estim.n_non_stationary];
    coordinates_transition_idio_cycles = coordinates_transition_stationary[1:end-estim.n_cycles];
    coordinates_transition_common_cycles = coordinates_transition_stationary[end-estim.n_cycles+1:end];

    # coordinates_transition_lagged and *_PPs
    coordinates_transition_lagged = sort(vcat(coordinates_transition_non_stationary, coordinates_transition_idio_cycles, [coordinates_transition_common_cycles .+ i for i=0:estim.lags-1]...));
    coordinates_transition_PPs = coordinates_transition_lagged .+ 1;

    #@infiltrate

    # `coordinates_transition_P0` identifies the entry in P0 that the cm-step should recompute
    coordinates_transition_P0 = findall(P0[:] .!= 0.0);

    # Comment this out if you want to keep the original diffuse initialisation for the non-stationary components
    # filter!(coordinate->coordinate.I[1] > size(P0_trends,1), coordinates_transition_P0);

    #=
    `coordinates_measurement_states` identifies the states for which the associated loadings are allowed to differ from zero.
    Note: the most compact version for this representation would be setdiff(coordinates_transition_lagged, coordinates_transition_drifts).
          However, this implementation of the DFM also includes the columns for the drifts, which always have zero loadings.
          This is purely a shortcut to simplify the implementation (estim.Γ_extended can be also used for the loadings cm step) and
          it does not impact the zero constraints described above.
    =#

    coordinates_measurement_states = copy(coordinates_transition_lagged);

    #@infiltrate

    # Convenient views for using sspace.B in the expected logliklihood and cm steps calculations
    B_star = @view sspace.B[:, coordinates_measurement_states];

    # Coordinates free parameters (B)
    cartesian_B = CartesianIndices(B_star);
    free_params_B_trends = zeros(estim.n, estim.n_trends) .== 1; # TBD: this is a bit tricky to relax since the coordinates in `coordinates_measurement_states` refer to the drifts for the I(2) trends - the direct Kitagawa representation for the smooth trend may help implementing it
    free_params_B_idio_cycles = zeros(estim.n, estim.n) .== 1;
    free_params_B_common_cycles = hcat(permutedims([estim.cycles_free_params[:,i] for i=1:estim.n_cycles, j=1:estim.lags])...);
    coordinates_free_params_B = cartesian_B[hcat(free_params_B_trends, free_params_B_idio_cycles, free_params_B_common_cycles)];
    @infiltrate
    
    # Convenient views for using sspace.C in the expected logliklihood and cm steps calculations
    C_star = @view sspace.C[coordinates_transition_current, coordinates_transition_lagged];

    # Coordinates free parameters (C)
    cartesian_C = CartesianIndices(C_star);
    free_params_C_trends = zeros(estim.n_non_stationary, estim.n_non_stationary) .== 1;
    free_params_C_idio_cycles = Matrix(I, estim.n, estim.n);
    free_params_C_common_cycles = cat(dims=[1,2], [ones(1, estim.lags) for i=1:estim.n_cycles]...) .== 1;
    coordinates_free_params_C = cartesian_C[cat(dims=[1,2], free_params_C_trends, free_params_C_idio_cycles, free_params_C_common_cycles)];

    @infiltrate

    # View on Q from DQD
    Q_view = @view sspace.DQD.data[coordinates_transition_current, coordinates_transition_current];

    @infiltrate
    
    # Return output
    return sspace, B_star, C_star, Q_view, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs, coordinates_transition_P0, coordinates_free_params_B, coordinates_free_params_C;
end

function initialise(estim::DFMSettings)
    return initialise(estim, estim.trends_skeleton);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Convergence check
--------------------------------------------------------------------------------------------------------------------------------
=#

function vec_ecm_params(estim::DFMSettings, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

    # Setup and memory pre-allocation
    counter = 1;
    output = zeros(length(coordinates_free_params_B) + length(coordinates_free_params_C) + size(Q_view,1));

    # Add parameters in B_star to output
    for ij in coordinates_free_params_B
        output[counter] = B_star[ij];
        counter += 1;
    end

    # Add parameters in C_star to output
    for ij in coordinates_free_params_C
        output[counter] = C_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for i in axes(Q_view,1)
        output[counter] = Q_view[i,i];
        counter += 1;
    end

    # Return output
    return output;
end

function vec_ecm_params!(estim::DFMSettings, output::FloatVector, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

    # Setup and memory pre-allocation
    counter = 1;

    # Add parameters in B_star to output
    for ij in coordinates_free_params_B
        output[counter] = B_star[ij];
        counter += 1;
    end

    # Add parameters in C_star to output
    for ij in coordinates_free_params_C
        output[counter] = C_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for i in axes(Q_view,1)
        output[counter] = Q_view[i,i];
        counter += 1;
    end
end

#=
--------------------------------------------------------------------------------------------------------------------------------
CM-step
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    cm_step!(estim::DFMSettings, sspace::KalmanSettings, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, F_raw::FloatMatrix, G::FloatMatrix, H_raw::FloatMatrix, M::FloatArray, N::Vector{SparseMatrixCSC{Float64, Int64}}, O::Array{VectorsArray{Float64},1}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

Run the CM-step for elastic-net DFMs.
"""
function cm_step!(estim::DFMSettings, sspace::KalmanSettings, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, F_raw::FloatMatrix, G::FloatMatrix, H_raw::FloatMatrix, M::FloatArray, N::Vector{SparseMatrixCSC{Float64, Int64}}, O::Array{VectorsArray{Float64},1}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

    # Make sure `F_raw` and `H_raw` are symmetric
    F = Symmetric(F_raw)::SymMatrix;
    H = Symmetric(H_raw)::SymMatrix;

    # Loop over the free parameters in B
    for ij in coordinates_free_params_B

        # Compute remaining sufficient statistics
        sum_numerator, sum_denominator, i, j = cm_step_time_loop(sspace, B_star, ij, N, O);

        # Compute new value for B_star[ij]
        B_star[ij] = soft_thresholding(M[ij] - sum_numerator, 0.5*estim.α*estim.ε*estim.Γ_extended[j, j]);
        B_star[ij] /= sum_denominator + (1-estim.α)*estim.ε*estim.Γ_extended[j, j];
    end

    # Copy the sspace.C components referring to the common cycles
    C_cycles_old = sspace.C[2*(estim.n_non_stationary+estim.n)+1:end, 2*(estim.n_non_stationary+estim.n)+1:end];
    C_cycles_new = @view sspace.C[2*(estim.n_non_stationary+estim.n)+1:end, 2*(estim.n_non_stationary+estim.n)+1:end];

    # Compute inverse of Q
    inv_Q = inv(Symmetric(Q_view));

    # Counter for common cycles and base coordinates
    n_previous_common_cycles = 0;
    base_coordinates_cycle = collect(1:estim.lags);

    # Loop over the free parameters
    for (position, ij) in enumerate(coordinates_free_params_C)

        # Coordinates
        i,j = ij.I;

        # Compute new value for sspace.C[ij]
        C_star[ij] = soft_thresholding(turbo_dot(inv_Q[:,i], G[:,j] - C_star*H[:,j]) + inv_Q[i,i]*C_star[ij]*H[j,j], 0.5*estim.α*estim.Γ_extended[j, j]);
        C_star[ij] /= inv_Q[i,i]*H[j,j] + (1-estim.α)*estim.Γ_extended[j, j];

        # Adjust autoregressive coefficient of the idiosyncratic cycles to enforce causality
        if position <= estim.n
            if abs(C_star[ij]) > 0.98
                C_star[ij] = sign(C_star[ij])*0.98;
            end

        # Adjust autoregressive coefficients of the common cycles to enforce causality
        elseif mod(position-estim.n, estim.lags) == 0
            
            # Position of the current cycle in `C_cycles_new` and `C_cycles_old`
            coordinates_current_cycle = base_coordinates_cycle .+ n_previous_common_cycles*(estim.lags+1);
            
            # Convenient views
            C_current_cycle_new = @view C_cycles_new[coordinates_current_cycle, coordinates_current_cycle];
            C_current_cycle_old = @view C_cycles_old[coordinates_current_cycle, coordinates_current_cycle];

            # Adjust coefficients (if needed)
            enforce_causality_and_invertibility!(C_current_cycle_new, C_current_cycle_old);

            # Update counter
            n_previous_common_cycles += 1;
        end
    end

    # CM-step for Q
    @infiltrate

    Q_cm_step = (F-G*C_star'-C_star*G'+C_star*H*C_star')/estim.T; # this matrix product could be improved, since it also considers the off-diagonal elements (which are not used in this cm-step)

    @infiltrate

    for i=1:estim.n_trends + estim.n + estim.n_cycles 
        Q_view[i,i] = Q_cm_step[i,i];
    end

    @infiltrate
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Model validation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    rescale_estim_params!(coordinates_params_rescaling::VectorsArray{Int64}, estim::DFMSettings, std_presample::FloatMatrix)

Rescale loadings associated to the trends.
"""
function rescale_estim_params!(coordinates_params_rescaling::VectorsArray{Int64}, estim::DFMSettings, std_presample::FloatMatrix)
    for (coordinate_std, coordinate_skeleton) in coordinates_params_rescaling
        estim.trends_skeleton[coordinate_skeleton] = 1/std_presample[coordinate_std];
    end
end

"""
    DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::Nothing, p::Int64, λ::Float64, α::Float64, β::Float64)
    DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::NamedTuple, p::Int64, λ::Float64, α::Float64, β::Float64)

Generate the relevant DFMSettings structure by using `model_inputs` and the candidate hyperparameters `p`, `λ`, `α`, `β`.
"""
DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::Nothing, p::Int64, λ::Float64, α::Float64, β::Float64) = DFMSettings(Y, p, model_args..., λ, α, β, verb=false);
DFMSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::NamedTuple, p::Int64, λ::Float64, α::Float64, β::Float64) = DFMSettings(Y, p, model_args..., λ, α, β; model_kwargs...);