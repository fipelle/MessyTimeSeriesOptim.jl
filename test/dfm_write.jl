"""
    estimation_input_test(estim::DFMSettings, benchmark_data::Tuple; is_stationary::Bool=false)

Check whether the settings in `estim` are equal to those in `benchmark_data`. If `is_stationary` is true skip the trend-related variables.

    estimation_input_test(estim_a::DFMSettings, estim_b::DFMSettings)

Check whether the settings in `estim_a` are equal to those in `estim_b`.
"""
function estimation_input_test(estim::DFMSettings, benchmark_data::Tuple; is_stationary::Bool=false)

    benchmark_Y, benchmark_n, benchmark_T, benchmark_lags, benchmark_n_trends, benchmark_n_drifts, benchmark_n_cycles,
        benchmark_n_non_stationary, benchmark_m, benchmark_trends_skeleton, benchmark_cycles_skeleton, benchmark_drifts_selection,
            benchmark_trends_free_params, benchmark_cycles_free_params, benchmark_λ, benchmark_α, benchmark_β, benchmark_Γ, 
                benchmark_Γ_idio, benchmark_Γ_extended_stationary, benchmark_Γ_extended_non_stationary, benchmark_ε, benchmark_tol, benchmark_max_iter, benchmark_prerun, benchmark_verb = benchmark_data;
    
    if is_stationary
        benchmark_n_trends = 0;
        benchmark_n_drifts = 0;
        benchmark_n_non_stationary = 0;
        benchmark_m = benchmark_n_cycles*benchmark_lags + benchmark_n;
        benchmark_trends_skeleton = nothing;
        benchmark_drifts_selection = nothing;
        benchmark_trends_free_params = nothing;
        benchmark_Γ_extended = benchmark_Γ_extended_stationary;
    else
        benchmark_Γ_extended = benchmark_Γ_extended_non_stationary;
    end

    return ~(false in [estim.Y === benchmark_Y;
                      estim.n == benchmark_n;
                      estim.T == benchmark_T;
                      estim.lags == benchmark_lags;
                      estim.n_trends == benchmark_n_trends;
                      estim.n_drifts == benchmark_n_drifts;
                      estim.n_cycles == benchmark_n_cycles;
                      estim.n_non_stationary == benchmark_n_non_stationary;
                      estim.m == benchmark_m;
                      estim.trends_skeleton == benchmark_trends_skeleton;
                      estim.cycles_skeleton == benchmark_cycles_skeleton;
                      estim.drifts_selection == benchmark_drifts_selection;
                      estim.trends_free_params == benchmark_trends_free_params;
                      estim.cycles_free_params == benchmark_cycles_free_params;
                      estim.λ == benchmark_λ;
                      estim.α == benchmark_α;
                      estim.β == benchmark_β;
                      estim.Γ == benchmark_Γ;
                      estim.Γ_idio == benchmark_Γ_idio;
                      estim.Γ_extended == benchmark_Γ_extended;
                      estim.ε == benchmark_ε;
                      estim.tol == benchmark_tol;
                      estim.max_iter == benchmark_max_iter;
                      estim.prerun == benchmark_prerun;
                      estim.verb == benchmark_verb]);
end

function estimation_input_test(estim_a::DFMSettings, estim_b::DFMSettings)

    return ~(false in [estim_a.Y === estim_b.Y;
                      estim_a.n == estim_b.n;
                      estim_a.T == estim_b.T;
                      estim_a.lags == estim_b.lags;
                      estim_a.n_trends == estim_b.n_trends;
                      estim_a.n_drifts == estim_b.n_drifts;
                      estim_a.n_cycles == estim_b.n_cycles;
                      estim_a.n_non_stationary == estim_b.n_non_stationary;
                      estim_a.m == estim_b.m;
                      estim_a.trends_skeleton == estim_b.trends_skeleton;
                      estim_a.cycles_skeleton == estim_b.cycles_skeleton;
                      estim_a.drifts_selection == estim_b.drifts_selection;
                      estim_a.trends_free_params == estim_b.trends_free_params;
                      estim_a.cycles_free_params == estim_b.cycles_free_params;
                      estim_a.λ == estim_b.λ;
                      estim_a.α == estim_b.α;
                      estim_a.β == estim_b.β;
                      estim_a.Γ == estim_b.Γ;
                      estim_a.Γ_idio == estim_b.Γ_idio;
                      estim_a.Γ_extended == estim_b.Γ_extended;
                      estim_a.ε == estim_b.ε;
                      estim_a.tol == estim_b.tol;
                      estim_a.max_iter == estim_b.max_iter;
                      estim_a.prerun == estim_b.prerun;
                      estim_a.verb == estim_b.verb]);
end

"""
    dfm_estimation_test(Y::JArray, lags::Int64, λ::Number, α::Number, β::Number, benchmark_data::Tuple)

Run basic tests on DFMSettings.
"""
function dfm_estimation_test(Y::JArray, lags::Int64, λ::Number, α::Number, β::Number, benchmark_data::Tuple)

    _, _, _, _, _, _, benchmark_n_cycles, _, _, benchmark_trends_skeleton, benchmark_cycles_skeleton, benchmark_drifts_selection,
        benchmark_trends_free_params, benchmark_cycles_free_params, _, _, _, _, _, _, _, _, _, _, _, _ = benchmark_data;

    constructor_settings = ((Y, lags, benchmark_n_cycles, λ, α, β), 
                            (Y, lags, benchmark_cycles_skeleton, benchmark_cycles_free_params, λ, α, β),
                            (Y, lags, benchmark_trends_skeleton, benchmark_cycles_skeleton, benchmark_drifts_selection, benchmark_trends_free_params, benchmark_cycles_free_params, λ, α, β));

    for (i, current_constructor_settings) in enumerate(constructor_settings)

        # Default settings

        estim_1 = DFMSettings(current_constructor_settings...);
        estim_2 = DFMSettings(current_constructor_settings..., ε=1e-4);
        estim_3 = DFMSettings(current_constructor_settings..., ε=1e-4, tol=1e-4);
        estim_4 = DFMSettings(current_constructor_settings..., ε=1e-4, tol=1e-4, max_iter=1000);
        estim_5 = DFMSettings(current_constructor_settings..., ε=1e-4, tol=1e-4, max_iter=1000, prerun=2);
        estim_6 = DFMSettings(current_constructor_settings..., ε=1e-4, tol=1e-4, max_iter=1000, prerun=2, verb=true);

        @test estimation_input_test(estim_1, benchmark_data, is_stationary=i<3);
        @test estimation_input_test(estim_1, estim_2);
        @test estimation_input_test(estim_1, estim_3);
        @test estimation_input_test(estim_1, estim_4);
        @test estimation_input_test(estim_1, estim_5);
        @test estimation_input_test(estim_1, estim_6);

        # Default settings (excl. ε)

        estim_7 = DFMSettings(current_constructor_settings..., ε=1e-8);
        estim_8 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-4);
        estim_9 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-4, max_iter=1000);
        estim_10 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-4, max_iter=1000, prerun=2);
        estim_11 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-4, max_iter=1000, prerun=2, verb=true);

        @test estimation_input_test(estim_7, estim_8);
        @test estimation_input_test(estim_7, estim_9);
        @test estimation_input_test(estim_7, estim_10);
        @test estimation_input_test(estim_7, estim_11);

        # Default settings (excl. ε, tol)

        estim_12 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16);
        estim_13 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=1000);
        estim_14 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=1000, prerun=2);
        estim_15 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=1000, prerun=2, verb=true);

        @test estimation_input_test(estim_12, estim_13);
        @test estimation_input_test(estim_12, estim_14);
        @test estimation_input_test(estim_12, estim_15);

        # Default settings (excl. ε, tol, max_iter)

        estim_16 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=10000);
        estim_17 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=10000, prerun=2);
        estim_18 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=10000, prerun=2, verb=true);

        @test estimation_input_test(estim_16, estim_17);
        @test estimation_input_test(estim_16, estim_18);

        # Default settings (excl. ε, tol, max_iter, prerun)

        estim_19 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=10000, prerun=3);
        estim_20 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=10000, prerun=3, verb=true);

        @test estimation_input_test(estim_19, estim_20);

        # Default settings (excl. ε, tol, max_iter, prerun, verb)
        
        estim_21 = DFMSettings(current_constructor_settings..., ε=1e-8, tol=1e-16, max_iter=10000, prerun=3, verb=false);
        
        @test estim_21.verb == false;
    end
end

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
    benchmark_baseline = read_test_input("./input/dfm/$(file_name)_baseline");
    benchmark_baseline_with_idio = read_test_input("./input/dfm/$(file_name)_baseline_with_idio");
    benchmark_sparse_dynamics = read_test_input("./input/dfm/$(file_name)_sparse_dynamics");
    benchmark_sparse_dynamics_with_idio = read_test_input("./input/dfm/$(file_name)_sparse_dynamics_with_idio");
    benchmark_full_sparse = read_test_input("./input/dfm/$(file_name)_full_sparse");
    benchmark_full_sparse_with_idio = read_test_input("./input/dfm/$(file_name)_full_sparse_with_idio");
    benchmark_messy = read_test_input("./input/dfm/$(file_name)_full_sparse_and_messy");
    benchmark_messy_with_idio = read_test_input("./input/dfm/$(file_name)_full_sparse_and_messy_with_idio");

    # Run tests
    @test norm(baseline-benchmark_baseline, Inf) <= 1e-8;
    @test norm(baseline_with_idio-benchmark_baseline_with_idio, Inf) <= 1e-8;
    @test norm(sparse_dynamics-benchmark_sparse_dynamics, Inf) <= 1e-8;
    @test norm(sparse_dynamics_with_idio-benchmark_sparse_dynamics_with_idio, Inf) <= 1e-8;
    @test norm(full_sparse-benchmark_full_sparse, Inf) <= 1e-8;
    @test norm(full_sparse_with_idio-benchmark_full_sparse_with_idio, Inf) <= 1e-8;
    @test norm(messy-benchmark_messy, Inf) <= 1e-8;
    @test norm(messy_with_idio-benchmark_messy_with_idio, Inf) <= 1e-8;
end

@testset "DFM basic functionalities" begin

    # Initialise data and benchmark data
    Y = [0.72 missing 1.86 missing missing 2.52 2.98 3.81 missing 4.36;
         0.95 0.70 missing missing missing missing 2.84 3.88 3.84 4.63];

    lags = 2;
    n = size(Y,1);
    T = size(Y,2);
    λ = 0.0;
    α = 0.5;
    β = 1.0;
    ε = 1e-4;
    tol = 1e-4;
    max_iter = 1000;
    prerun = 2;
    verb = true;

    # Advanced benchmark components
    trends_skeleton = Matrix(1.0I, n, n);
    cycles_skeleton = hcat([1; 2*ones(n-1)]);
    drifts_selection = ones(n) .== 1;
    trends_free_params = trends_skeleton .== 2;
    cycles_free_params = cycles_skeleton .> 1;
    Γ = build_Γ(1, lags, λ, β);
    Γ_idio = build_Γ(1, 1, λ, β)[1];
    Γ_extended_stationary = cat(dims=[1,2], [Γ_idio for i=1:n]..., Γ) |> Array |> Diagonal;
    Γ_extended_non_stationary = cat(dims=[1,2], Diagonal(zeros(2*n, 2*n)), [Γ_idio for i=1:n]..., Γ) |> Array |> Diagonal;

    # Setup benchmark data
    benchmark_data = (Y, n, T, lags, 2, 2, 1, 4, 2*n+n+lags, trends_skeleton, cycles_skeleton, drifts_selection, trends_free_params, cycles_free_params, λ, α, β, Γ, Γ_idio, Γ_extended_stationary, Γ_extended_non_stationary, ε, tol, max_iter, prerun, verb);

    # Run tests
    dfm_estimation_test(Y, lags, λ, α, β, benchmark_data);
end

@testset "DFM simulations: MLE" begin
    dfm_simulation_tests("mle", 0.0, 0.0, 1.0);
end

@testset "DFM simulations: penalised MLE" begin
    models_triplets = Dict("ridge" => (10.0, 0.0, 1.0), "lasso" => (10.0, 1.0, 1.0), "en" => (10.0, 0.5, 1.0));
    
    for (key, value) in models_triplets
        dfm_simulation_tests(key, value...);
    end
end