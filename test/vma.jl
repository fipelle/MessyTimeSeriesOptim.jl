"""
    estimation_input_test(estim::VMASettings, benchmark_data::Tuple)

Check whether the settings in `estim` are equal to those in `benchmark_data`.

    estimation_input_test(estim_a::VMASettings, estim_b::VMASettings)

Check whether the settings in `estim_a` are equal to those in `estim_b`.
"""
function estimation_input_test(estim::VMASettings, benchmark_data::Tuple)

    benchmark_Y, benchmark_n, benchmark_T, benchmark_r, benchmark_nr, benchmark_m, benchmark_λ, 
        benchmark_α, benchmark_β, benchmark_Γ, benchmark_ε, benchmark_tol, benchmark_max_iter, 
            benchmark_prerun, benchmark_verb = benchmark_data;
    
    return ~(false in [estim.Y === benchmark_Y;
                      estim.n == benchmark_n;
                      estim.T == benchmark_T;
                      estim.r == benchmark_r;
                      estim.nr == benchmark_nr;
                      estim.m == benchmark_m;
                      estim.λ == benchmark_λ;
                      estim.α == benchmark_α;
                      estim.β == benchmark_β;
                      estim.Γ == benchmark_Γ;
                      estim.ε == benchmark_ε;
                      estim.tol == benchmark_tol;
                      estim.max_iter == benchmark_max_iter;
                      estim.prerun == benchmark_prerun;
                      estim.verb == benchmark_verb]);
end

function estimation_input_test(estim_a::VMASettings, estim_b::VMASettings)
    
    return ~(false in [estim_a.Y === estim_b.Y;
                      estim_a.n == estim_b.n;
                      estim_a.T == estim_b.T;
                      estim_a.r == estim_b.r;
                      estim_a.nr == estim_b.nr;
                      estim_a.m == estim_b.m;
                      estim_a.λ == estim_b.λ;
                      estim_a.α == estim_b.α;
                      estim_a.β == estim_b.β;
                      estim_a.Γ == estim_b.Γ;
                      estim_a.ε == estim_b.ε;
                      estim_a.tol == estim_b.tol;
                      estim_a.max_iter == estim_b.max_iter;
                      estim_a.prerun == estim_b.prerun;
                      estim_a.verb == estim_b.verb]);
end

"""
    vma_estimation_test(Y::JArray, r::Int64, λ::Number, α::Number, β::Number, benchmark_data::Tuple)

Run basic tests on VMASettings.
"""
function vma_estimation_test(Y::JArray, r::Int64, λ::Number, α::Number, β::Number, benchmark_data::Tuple)

    #=
    Tests on VMASettings
    =#

    # First group of settings

    estim1 = VMASettings(Y, r, λ, α, β);
    estim2 = VMASettings(Y, r, λ, α, β, ε=1e-4);
    estim3 = VMASettings(Y, r, λ, α, β, ε=1e-4, tol=1e-4);
    estim4 = VMASettings(Y, r, λ, α, β, ε=1e-4, tol=1e-4, max_iter=1000);
    estim5 = VMASettings(Y, r, λ, α, β, ε=1e-4, tol=1e-4, max_iter=1000, prerun=2);
    estim6 = VMASettings(Y, r, λ, α, β, ε=1e-4, tol=1e-4, max_iter=1000, prerun=2, verb=true);
    
    @test estimation_input_test(estim1, benchmark_data);
    @test estimation_input_test(estim1, estim2);
    @test estimation_input_test(estim1, estim3);
    @test estimation_input_test(estim1, estim4);
    @test estimation_input_test(estim1, estim5);
    @test estimation_input_test(estim1, estim6);

    # Second group of settings

    estim1b = VMASettings(Y, r, λ, α, β, ε=1e-8);
    estim2b = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-4);
    estim3b = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-4, max_iter=1000);
    estim4b = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-4, max_iter=1000, prerun=2);
    estim5b = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-4, max_iter=1000, prerun=2, verb=true);
    
    @test estimation_input_test(estim1b, estim2b);
    @test estimation_input_test(estim1b, estim3b);
    @test estimation_input_test(estim1b, estim4b);
    @test estimation_input_test(estim1b, estim5b);

    # Third group of settings

    estim1c = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16);
    estim2c = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=1000);
    estim3c = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=1000, prerun=2);
    estim4c = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=1000, prerun=2, verb=true);
    
    @test estimation_input_test(estim1c, estim2c);
    @test estimation_input_test(estim1c, estim3c);
    @test estimation_input_test(estim1c, estim4c);

    # Fourth group of settings

    estim1d = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=10000);
    estim2d = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=10000, prerun=2);
    estim3d = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=10000, prerun=2, verb=true);
    
    @test estimation_input_test(estim1d, estim2d);
    @test estimation_input_test(estim1d, estim3d);

    # Fifth group of settings

    estim1e = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=10000, prerun=3);
    estim2e = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=10000, prerun=3, verb=true);
    
    @test estimation_input_test(estim1e, estim2e);

    # Sixth group of settings

    estim1f = VMASettings(Y, r, λ, α, β, ε=1e-8, tol=1e-16, max_iter=10000, prerun=3, verb=false);
    
    @test estim1f.verb == false;

    #= 
    Set default settings
    =#

    estim = estim1;

    #=
    Tests on initialisation
    =#

    sspace, B_star, C_star, Q_view, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs, coordinates_transition_P0, coordinates_free_params_B, coordinates_free_params_C = MessyTimeSeriesOptim.initialise(estim);

    @test sspace.B[:, 1:estim.n] == Matrix(1.0I, estim.n, estim.n);
    @test sspace.C[1:estim.n, :] == zeros(estim.n, estim.n+estim.m);

    #=
    Tests on ksmoother_ecm!
    =#

    # Run Kalman filter
    status = SizedKalmanStatus(sspace.Y.T);
    kfilter_full_sample!(sspace, status);

    # Compute Kalman stats from the ksmoother function
    X_sm, P_sm, X0_sm, P0_sm = ksmoother(sspace, status);
    
    # Memory pre-allocation
    smoother_arrays_benchmark = MessyTimeSeriesOptim.SmootherArrays(estim, sspace, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);

    # Compute F_benchmark, G_benchmark and H_benchmark
    for t=length(X_sm)-1:-1:1
        copyto!(smoother_arrays_benchmark.Xs_leading, X_sm[t+1]);
        copyto!(smoother_arrays_benchmark.Ps_leading, P_sm[t+1]);
        Xs = X_sm[t];
        Ps = P_sm[t];
        MessyTimeSeriesOptim.update_ecm_stats_transition!(smoother_arrays_benchmark.F, smoother_arrays_benchmark.G, estim, smoother_arrays_benchmark, Xs, Ps, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);
    end

    copyto!(smoother_arrays_benchmark.Xs_leading, X_sm[1]);
    copyto!(smoother_arrays_benchmark.Ps_leading, P_sm[1]);
    MessyTimeSeriesOptim.update_ecm_stats_transition!(smoother_arrays_benchmark.F, smoother_arrays_benchmark.G, estim, smoother_arrays_benchmark, X0_sm, P0_sm, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);

    # Compute Kalman stats from the ksmoother_ecm! function
    smoother_arrays = MessyTimeSeriesOptim.SmootherArrays(estim, sspace, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);
    MessyTimeSeriesOptim.ksmoother_ecm!(estim, sspace, status, smoother_arrays, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs, coordinates_transition_P0);

    # Tests
    @test smoother_arrays.F == smoother_arrays_benchmark.F;
    @test smoother_arrays.G == smoother_arrays_benchmark.G;
    @test smoother_arrays.H == smoother_arrays_benchmark.H;
    @test sspace.X0 == X0_sm;
    @test sspace.P0 == P0_sm;
end

"""
    naive_vma_simulation(true_coefficient::FloatMatrix, n::Int64, T::Int64, r::Int64, λ::Number, α::Number, β::Number; no_sim::Int64=500, verb::Bool=false)

Run a VMA(r) naive simulation.
"""
function naive_vma_simulation(true_coefficient::FloatMatrix, n::Int64, T::Int64, r::Int64, λ::Number, α::Number, β::Number; no_sim::Int64=500, verb::Bool=false)

    # Run simulation
    rng = StableRNG(1);
    B_sim = zeros(n, n*r, no_sim);
    params_accuracy = zeros(no_sim);
    oos_fc_error = zeros(no_sim);

    for sim=1:no_sim

        # Simulate data with mean zero and unit standard deviation
        v = randn(rng, n, T+r);
        Y = zeros(n, T+r);
        for t=1:T+r
            if t <= r
                Y[:,t] .= 0.0;
            else
                Y[:,t] = true_coefficient*vcat([v[:,t-lag][:] for lag=1:r]...) + v[:,t];
            end
        end
        Y = standardise(Y[:, r+1:end]);

        # Estimate
        t0 = floor(T/2) |> Int64;
        estim = VMASettings(Y[:,1:t0], r, λ, α, β, verb=false);
        sspace = ecm(estim, output_sspace_data=Y);

        # Run Kalman filter
        status = kfilter_full_sample(sspace);

        # Forecast
        forecast = mapreduce(Xt -> sspace.B*Xt, hcat, status.history_X_prior);

        # Store estimated parameters
        B_sim[:,:,sim] = sspace.B[1:n, n+1:n+n*r];

        # Store params_accuracy
        difference_B_truth = @views B_sim[:,:,sim] - true_coefficient;
        params_accuracy[sim] = sqrt(tr(difference_B_truth'*difference_B_truth));

        # Store oos_fc_error
        oos_fc_error[sim] = @views mean(mean(abs.(Y[:,t0+1:end] - forecast[:,t0+1:end]), dims=2)./mean(abs.(Y[:,t0+1:end]), dims=2));
    end

    if verb == true
        println("=============================================================================================================================================")
        println("True parameters:")
        println(true_coefficient);
        println("=============================================================================================================================================")
        println("Estimated parameters with λ=$(λ), α=$(α), and β=$(β):")
        println(round.(median(B_sim, dims=3)[:,:], digits=5));
        println("=============================================================================================================================================")
        println("");
    end

    return B_sim, params_accuracy, oos_fc_error;
end

"""
    vma_simulation_tests(model_id::String, λ::Number, α::Number, β::Number)

Default tests based on VMA simulations.
"""
function vma_simulation_tests(model_id::String, λ::Number, α::Number, β::Number)

    # Run simulations
    _, bivariate_vma1_full, _ = naive_vma_simulation([0.3 -0.2; -0.2 0.5], 2, 200, 1, λ, α, β, no_sim=10);
    _, bivariate_vma2_full, _ = naive_vma_simulation([0.5 0.4 0.2 0.1; 0.3 -0.2 0.1 0.1], 2, 200, 2, λ, α, β, no_sim=10);
    _, bivariate_vma2_sparse, _ = naive_vma_simulation([0.3 0.0 0.0 0.0; 0.0 -0.1 0.0 0.0], 2, 200, 2, λ, α, β, no_sim=10);
    _, trivariate_vma2_groups, _ = naive_vma_simulation([0.5 0.4 0.0 0.2 0.1 0.0; 0.3 -0.2 0.0 0.1 0.1 0.0; 0.0 0.0 0.9 0.0 0.0 0.0], 3, 200, 2, λ, α, β, no_sim=10);

    # Load benchmarks
    benchmark_bivariate_vma1_full = read_test_input("./input/vma/$(model_id)_bivariate_vma1_full");
    benchmark_bivariate_vma2_full = read_test_input("./input/vma/$(model_id)_bivariate_vma2_full");
    benchmark_bivariate_vma2_sparse = read_test_input("./input/vma/$(model_id)_bivariate_vma2_sparse");
    benchmark_trivariate_vma2_groups = read_test_input("./input/vma/$(model_id)_trivariate_vma2_groups");

    # Perform tests
    @test norm(bivariate_vma1_full-benchmark_bivariate_vma1_full, Inf) <= 1e-8;
    @test norm(bivariate_vma2_full-benchmark_bivariate_vma2_full, Inf) <= 1e-8;
    @test norm(bivariate_vma2_sparse-benchmark_bivariate_vma2_sparse, Inf) <= 1e-8;
    @test norm(trivariate_vma2_groups-benchmark_trivariate_vma2_groups, Inf) <= 1e-8;
end

@testset "VMA basic functionalities" begin

    # Initialise data and state-space parameters
    Y = [0.72 missing 1.86 missing missing 2.52 2.98 3.81 missing 4.36;
         0.95 0.70 missing missing missing missing 2.84 3.88 3.84 4.63];

    r = 2;
    nr = 4;
    m = 6;
    λ = 0.0;
    α = 0.5;
    β = 1.0;
    ε = 1e-4;
    tol = 1e-4;
    max_iter = 1000;
    prerun = 2;
    verb = true;

    Γ_1 = build_Γ(2, r, λ, β);
    Γ_2 = build_Γ(2, r, λ+1, β);
    Γ_3 = build_Γ(2, r, λ+1, β+1);

    # Benchmark data
    benchmark_data_1 = (Y, size(Y,1), size(Y,2), r, nr, m, λ, α, β, Γ_1, ε, tol, max_iter, prerun, verb);
    benchmark_data_2 = (Y, size(Y,1), size(Y,2), r, nr, m, λ+1, α, β, Γ_2, ε, tol, max_iter, prerun, verb);
    benchmark_data_3 = (Y, size(Y,1), size(Y,2), r, nr, m, λ+1, α, β+1, Γ_3, ε, tol, max_iter, prerun, verb);

    # Run tests
    vma_estimation_test(Y, r, λ, α, β, benchmark_data_1);
    vma_estimation_test(Y, r, λ+1, α, β, benchmark_data_2);
    vma_estimation_test(Y, r, λ+1, α, β+1, benchmark_data_3);
end

@testset "VMA simulations: MLE" begin
    vma_simulation_tests("mle", 0.0, 0.0, 1.0);
end

@testset "VMA simulations: penalised MLE" begin

    models_triplets = Dict("ridge" => (10.0, 0.0, 1.0), "lasso" => (10.0, 1.0, 1.0), "en" => (10.0, 0.5, 1.0));
    
    for (key, value) in models_triplets
        vma_simulation_tests(key, value...);
    end
end