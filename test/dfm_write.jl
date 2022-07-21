include("./tools.jl");

"""
    naive_dfm_simulation(true_loadings::FloatMatrix, true_dynamics::FloatMatrix, true_idio_dynamics::FloatVector, n::Int64, n_cycles::Int64, T::Int64, lags::Int64, λ::Number, α::Number, β::Number; no_sim::Int64=500)

Run a DFM naive simulation.
"""
function naive_dfm_simulation(true_loadings::FloatMatrix, true_dynamics::FloatMatrix, true_idio_dynamics::FloatVector, n::Int64, n_cycles::Int64, T::Int64, lags::Int64, λ::Number, α::Number, β::Number; no_sim::Int64=500)
    
    if n_cycles > 1
        error("This function is not compatible with more than 1 common cycle");
    end

    # Initial setup
    trends_skeleton = Matrix(1.0I, n, n);
    cycles_skeleton = hcat([[zeros(i-1); 1; 2*ones(n-i)] for i=1:n_cycles]...);
    cycles_free_params = cycles_skeleton .> 1;
    trends_free_params = zeros(size(trends_skeleton)) .== 1;
    drifts_selection = ones(n) .== 1;

    # Run simulation
    rng = StableRNG(1);
    mase_common_cycles = zeros(no_sim);
    mase_total_cycles = zeros(no_sim);

    for sim=1:no_sim

        drifts = randn(rng, n);
        trends = zeros(n, T+lags);
        idio_cycles = zeros(n, T+lags);
        common_cycles = zeros(n_cycles, T+lags);
        Y = zeros(n, T+lags);

        # Run simulation
        for t=1:T+lags
            if t <= lags
                trends[:,t] .= 0;
                idio_cycles[:,t] .= 0;
                common_cycles[:,t] .= 0;
            else
                trends[:,t] = drifts + trends[:,t-1] + randn(rng, n);
                idio_cycles[:,t] = true_idio_dynamics .* idio_cycles[:,t-1] + randn(rng, n);
                common_cycles[:,t] = true_dynamics*vcat([common_cycles[:, t-lag][:] for lag=1:lags]...) + randn(rng, n_cycles);
                Y[:,t] = trends[:,t] + idio_cycles[:,t] + true_loadings*vcat([common_cycles[:,t-lag][:] for lag=0:lags-1]...);
            end
        end

        # Cut data and unobserved components
        trends = trends[:,lags+1:end];
        idio_cycles = idio_cycles[:,lags+1:end];
        common_cycles = common_cycles[:,lags+1:end];
        Y = Y[:,lags+1:end];

        # Standardise data and setup estimation
        std_factor = std(Y,dims=2);
        Y = Y./std_factor;
        estim = DFMSettings(Y, lags, trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params, λ, α, β, verb=false);

        # Estimate
        sspace = ecm(estim);

        # Run Kalman filter
        status = SizedKalmanStatus(sspace.Y.T);
        kfilter_full_sample!(sspace, status);

        X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status);
        smoothed_states = hcat([X_sm[i] for i=1:length(X_sm)]...);

        # Extract relevant cycles
        total_cycles = (Y .* std_factor) - trends;
        common_cycles_projections = total_cycles - idio_cycles;
        estimated_common_cycle_projections = (sspace.B[:, 4*n+2*n+1:end-1]*smoothed_states[4*n+2*n+1:end-1, :]) .* std_factor;
        estimated_total_cycles = (sspace.B[:, 4*n+2*n+1:end-1]*smoothed_states[4*n+2*n+1:end-1, :] + smoothed_states[4*n+1:2:4*n+2*n, :]) .* std_factor;

        # Compute mean absolute scaled error (note that the denominator is different compared to the usual formula, since cycle is causal)
        mase_common_cycles[sim] = mean(mean(abs.(common_cycles_projections - estimated_common_cycle_projections), dims=2)./mean(abs.(common_cycles_projections), dims=2));
        mase_total_cycles[sim] = mean(mean(abs.(total_cycles - estimated_total_cycles), dims=2)./mean(abs.(total_cycles), dims=2));
    end

    return mase_common_cycles, mase_total_cycles;
end

function write_test_input(path, content)
    open("$path.txt", "a") do io
        print(io, content)
    end
end

"""
    dfm_simulation_tests(file_name::String, λ::Number, α::Number, β::Number)

Default tests based on DFM simulations.
"""
function dfm_simulation_tests(file_name::String, λ::Number, α::Number, β::Number)

    # Initial settings
    n = 4;
    r = 1;
    T = 100;
    lags = 4;

    # Dense settings
    loadings_full = [1.00  0.00 0.00 0.00; 
                     -0.06 -0.15 0.03 0.20;
                     0.50 -0.20 -0.05 -0.05;
                     0.75  0.50 -0.04 -0.65];

    dynamics_full = [1.5 -0.55 0.10 -0.10];

    # Sparse settings
    loadings_sparse = [1.00   0.00 0.00  0.00; 
                       -0.06 -0.00 0.03  0.20;
                       0.50   0.00 -0.05 0.00;
                       0.00   0.50 0.00  0.00];
    
    dynamics_sparse = [0.6 0.0 -0.4 0.0];

    # Messy loadings settings
    rng = StableRNG(10);
    loadings_messy = copy(loadings_sparse);
    loadings_messy .+= 0.1*randn(rng, n, lags);

    # Idiosyncratic coefficients
    idio_vect = 0.2*ones(n);

    # Baseline
    _, baseline = naive_dfm_simulation(loadings_full, dynamics_full, zeros(n), n, 1, T, lags, λ, α, β, no_sim=10);
    _, baseline_with_idio = naive_dfm_simulation(loadings_full, dynamics_full, idio_vect, n, 1, T, lags, λ, α, β, no_sim=10);

    # Sparse dynamics
    _, sparse_dynamics = naive_dfm_simulation(loadings_full, dynamics_sparse, zeros(n), n, 1, T, lags, λ, α, β, no_sim=10);
    _, sparse_dynamics_with_idio = naive_dfm_simulation(loadings_full, dynamics_sparse, idio_vect, n, 1, T, lags, λ, α, β, no_sim=10);

    # Full sparse
    _, full_sparse = naive_dfm_simulation(loadings_sparse, dynamics_sparse, zeros(n), n, 1, T, lags, λ, α, β, no_sim=10);
    _, full_sparse_with_idio = naive_dfm_simulation(loadings_sparse, dynamics_sparse, idio_vect, n, 1, T, lags, λ, α, β, no_sim=10);

    # Messy
    _, messy = naive_dfm_simulation(loadings_messy, dynamics_sparse, zeros(n), n, 1, T, lags, λ, α, β, no_sim=10);
    _, messy_with_idio = naive_dfm_simulation(loadings_messy, dynamics_sparse, idio_vect, n, 1, T, lags, λ, α, β, no_sim=10);

    # Load benchmark data
    write_test_input("./input/dfm/$(file_name)_baseline", baseline);
    write_test_input("./input/dfm/$(file_name)_baseline_with_idio", baseline_with_idio);
    write_test_input("./input/dfm/$(file_name)_sparse_dynamics", sparse_dynamics);
    write_test_input("./input/dfm/$(file_name)_sparse_dynamics_with_idio", sparse_dynamics_with_idio);
    write_test_input("./input/dfm/$(file_name)_full_sparse", full_sparse);
    write_test_input("./input/dfm/$(file_name)_full_sparse_with_idio", full_sparse_with_idio);
    write_test_input("./input/dfm/$(file_name)_full_sparse_and_messy", messy);
    write_test_input("./input/dfm/$(file_name)_full_sparse_and_messy_with_idio", messy_with_idio);
end

dfm_simulation_tests("mle", 0.0, 0.0, 1.0);

models_triplets = Dict("ridge" => (10.0, 0.0, 1.0), "lasso" => (10.0, 1.0, 1.0), "en" => (10.0, 0.5, 1.0));

for (key, value) in models_triplets
    dfm_simulation_tests(key, value...);
end
