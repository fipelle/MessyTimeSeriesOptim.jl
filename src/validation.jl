"""
    rescale_estim_params!(coordinates_params_rescaling::Nothing, estim::EstimSettings, std_presample::FloatMatrix)

Default implementation for a generic `estim` does nothing.
"""
rescale_estim_params!(coordinates_params_rescaling::Nothing, estim::EstimSettings, std_presample::FloatMatrix) = nothing;

"""
    compute_loss_weights(custom_weights::FloatVector, n::Int64)

Use `custom_weights`.

    compute_loss_weights(custom_weights::Nothing, n::Int64)

Use equal weights.
"""
compute_loss_weights(custom_weights::FloatVector, n::Int64) = custom_weights;
compute_loss_weights(custom_weights::Nothing, n::Int64) = ones(n) ./ n;

"""
    compute_loss(weighted_se::AbstractArray{Float64})
    compute_loss(weighted_se::AbstractArray{Missing})
    compute_loss(weighted_se::AbstractArray{Union{Float64, Missing}})

Compute loss function. Return `[error inactive_sample]`. 
 
`inactive_sample` is 1 when the data is entirely missing (0 otherwise).
"""
compute_loss(weighted_se::AbstractArray{Float64}) = [sum(mean(weighted_se, dims=2)), 0.0];
compute_loss(weighted_se::AbstractArray{Missing}) = [0.0, 1.0];

function compute_loss(weighted_se::AbstractArray{Union{Float64, Missing}})

    # Setup
    n, T = size(weighted_se);

    # Stop if the weighted squared errors are entirely missing
    if count(ismissing, weighted_se) == n*T
        return [0.0, 1.0];

    # Standard run
    else
        return [sum(sum_skipmissing(weighted_se))/T, 0.0];
    end
end

"""
    fc_err(validation_settings::ValidationSettings, p::Int64, λ::Number, α::Number, β::Number)

Compute the in-sample / out-of-sample error associated with the candidate hyperparameters

# Arguments
- `validation_settings`: ValidationSettings struct
- `p`: (candidate) number of lags in the vector autoregression
- `λ`: (candidate) overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: (candidate) weight associated to the LASSO component of the elastic-net penalty
- `β`: (candidate) additional shrinkage for distant lags (p>1)
- `jth_jackknife_data`: j-th jackknife sample (default: nothing)

# References
Pellegrino (2022)
"""
function fc_err(validation_settings::ValidationSettings, p::Int64, λ::Number, α::Number, β::Number; jackknife_data::Union{JArray{Float64}, Nothing}=nothing, jackknife_sample_id::Union{Int64, Nothing}=nothing);

    # In-sample error
    if validation_settings.err_type == 1
        t0 = validation_settings.T;
    
    # Out-of-sample based error
    else
        t0 = validation_settings.t0;
    end

    # Jackknife out-of-sample
    if validation_settings.err_type > 2
        data = @view jackknife_data[:, :, jackknife_sample_id];

    # Standard in-sample or out-of-sample
    else
        data = validation_settings.Y;
    end

    # Standardise data
    data_presample_view = @view data[:, 1:t0];

    # Stop if the estimation sample is entirely missing
    if count(ismissing, data_presample_view) == validation_settings.n*t0
        return [0.0, 1.0];

    # Standard run
    else
        if validation_settings.is_stationary
            mean_presample = mean_skipmissing(data_presample_view);
            std_presample = std_skipmissing(data_presample_view);
            Y = (data .- mean_presample) ./ std_presample;
        else
            std_presample = std_skipmissing(diff(data_presample, dims=2));
            Y = data ./ std_presample;
        end

        estim = validation_settings.model_struct(Y[:, 1:t0], validation_settings.model_args, validation_settings.model_kwargs, p, λ, α, β);
        rescale_estim_params!(validation_settings.coordinates_params_rescaling, estim, std_presample);

        # Estimate model
        sspace = ecm(estim, output_sspace_data=Y);

        # Run Kalman filter
        status = kfilter_full_sample(sspace);

        # Forecast
        X_prior = mapreduce(Xt -> sspace.B*Xt, hcat, status.history_X_prior);

        # Apply original scaling
        if validation_settings.is_stationary
            forecast = (X_prior .* std_presample) .+ mean_presample;
        else
            forecast = X_prior .* std_presample;
        end

        # Compute weights
        w = compute_loss_weights(validation_settings.weights, validation_settings.n);

        # In-sample error
        if validation_settings.err_type == 1
            weighted_se = w .* (data .- forecast).^2;

        # Out-of-sample error
        else
            weighted_se = @views w .* (data[:, t0+1:end] .- forecast[:, t0+1:end]).^2;
        end

        # Return output
        return compute_loss(weighted_se);
    end
end

"""
    jackknife_err(validation_settings::ValidationSettings, jackknife_data::JArray{Float64, 3}, p::Int64, λ::Number, α::Number, β::Number)

Return the jackknife out-of-sample error.

# Arguments
- `validation_settings`: ValidationSettings struct
- `jackknife_data`: jackknife partitions
- `p`: (candidate) number of lags in the vector autoregression
- `λ`: (candidate) overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: (candidate) weight associated to the LASSO component of the elastic-net penalty
- `β`: (candidate) additional shrinkage for distant lags (p>1)

# References
Pellegrino (2022)
"""
function jackknife_err(validation_settings::ValidationSettings, jackknife_data::JArray{Float64, 3}, p::Int64, λ::Number, α::Number, β::Number)

    # Error management
    if validation_settings.err_type <= 2
        error("Wrong err_type for jackknife_err!");
    end

    # Number of jackknife samples
    samples = size(jackknife_data, 3);

    # Compute jackknife loss
    verb_message(validation_settings.verb_estim, "jackknife_err > running $samples iterations on $(nworkers()) workers");

    output_fc_err = @sync @distributed (+) for j=1:samples
        fc_err(validation_settings, p, λ, α, β, jackknife_data=jackknife_data, jackknife_sample_id=j);
    end

    # Compute average jackknife loss
    loss, inactive_samples = output_fc_err;
    if samples == inactive_samples
        error("All samples are inactive! Check the initial settings or try a different randomisation method.");
    end
    loss *= 1/(samples-inactive_samples);

    # Return output
    return loss;
end

"""
    select_hyperparameters(validation_settings::ValidationSettings, γ_grid::HyperGrid)

Select the tuning hyper-parameters for the elastic-net vector autoregression.

# Arguments
- `validation_settings`: ValidationSettings struct
- `γ_grid`: HyperGrid struct

# References
Pellegrino (2022)
"""
function select_hyperparameters(validation_settings::ValidationSettings, γ_grid::HyperGrid, seed::Int64=1)

    # Check inputs
    check_bounds(validation_settings.n, 2); # It supports only multivariate models (for now ...)

    if length(γ_grid.p) != 2 || length(γ_grid.λ) != 2 || length(γ_grid.α) != 2 || length(γ_grid.β) != 2
        error("The grids include more or less than two entries. They must include only the lower and upper bounds for the grids!")
    end

    check_bounds(γ_grid.p[2], γ_grid.p[1]);
    check_bounds(γ_grid.λ[2], γ_grid.λ[1]);
    check_bounds(γ_grid.α[2], γ_grid.α[1]);
    check_bounds(γ_grid.β[2], γ_grid.β[1]);
    check_bounds(γ_grid.p[1], 1);
    check_bounds(γ_grid.λ[1], 0);
    check_bounds(γ_grid.α[1], 0, 1);
    check_bounds(γ_grid.α[2], 0, 1);
    check_bounds(γ_grid.β[1], 1);

    # Memory pre-allocation for random search
    rng = StableRNG(seed);
    errors = zeros(γ_grid.draws);
    candidates = zeros(4, γ_grid.draws);

    for draw=1:γ_grid.draws
        candidates[:,draw] = [rand(rng, γ_grid.p[1]:γ_grid.p[2]),
                              rand(rng, Uniform(γ_grid.λ[1], γ_grid.λ[2])),
                              rand(rng, Uniform(γ_grid.α[1], γ_grid.α[2])),
                              rand(rng, Uniform(γ_grid.β[1], γ_grid.β[2]))];
    end

    # Setup logger path when `validation_settings.log_folder_path` is specified
    if (validation_settings.verb == true) && ~isnothing(validation_settings.log_folder_path)
        io = open("$(validation_settings.log_folder_path)/status_err_type_$(validation_settings.err_type).txt", "w+");
        global_logger(ConsoleLogger(io));
    end

    # Generate partitions for the block jackknife out-of-sample
    if validation_settings.err_type == 3
        jackknife_data = block_jackknife(validation_settings.Y, validation_settings.subsample);

    # Generate partitions for the artificial jackknife
    elseif validation_settings.err_type == 4
        jackknife_data = artificial_jackknife(validation_settings.Y, validation_settings.subsample, validation_settings.max_samples);
    end

    for iter=1:γ_grid.draws

        # Retrieve candidate hyperparameters
        p, λ, α, β = candidates[:,iter];
        p = Int64(p);

        # Update log
        if validation_settings.verb == true
            @info "$(round(now(), Dates.Second(1))) select_hyperparameters (error estimator $(validation_settings.err_type)) > running iteration $iter (out of $(γ_grid.draws)), γ=($(round(p,digits=3)), $(round(λ,digits=3)), $(round(α,digits=3)), $(round(β,digits=3)))";
            if ~isnothing(validation_settings.log_folder_path)
                flush(io);
            end
        end

        #=
        Some candidate hyperparameters might lead to numerical instabilities. This generally happens when:
        - the candidates are extreme with respect to the model of interest
        - validation_settings.subsample is high relative to the sample size and there is not enough shrinkage

        In simulation, this issue was mostly observed with the block-jackknife.

        A priori, there is not an obvious way to construct a grid of candidates that does not result in errors in the ECM.
        The fc_err function handles these numerical issues on a case-by-case basis.
        =#

        # In-sample or standard out-of-sample
        if validation_settings.err_type <= 2
            errors[iter], inactive_sample = fc_err(validation_settings, p, λ, α, β);
            if inactive_sample == 1
                error("The estimation / validation sample is a matrix of missings!");
            end

        # Jackknife out-of-sample
        else
            errors[iter] = jackknife_err(validation_settings, jackknife_data, p, λ, α, β);
        end
    end

    # Close log
    if (validation_settings.verb == true) && ~isnothing(validation_settings.log_folder_path)
        close(io);
    end

    # Return output
    return candidates, errors;
end
