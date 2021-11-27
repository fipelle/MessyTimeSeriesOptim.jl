"""
    validation_input_test(validation::ValidationSettings, benchmark_data::Tuple)

Check whether the settings in `validation` are equal to those in `benchmark_data`.

    validation_input_test(validation_a::ValidationSettings, validation_b::ValidationSettings)

Check whether the settings in `validation_a` are equal to those in `validation_b`.
"""
function validation_input_test(validation::ValidationSettings, benchmark_data::Tuple)

    benchmark_err_type, benchmark_Y, benchmark_n, benchmark_T, benchmark_is_stationary, benchmark_model_struct, benchmark_model_args, benchmark_verb,
        benchmark_verb_estim, benchmark_weights, benchmark_t0, benchmark_subsample, benchmark_max_samples, benchmark_log_folder_path = benchmark_data; # TBD: add test on kwargs

    return ~(false in [validation.err_type == benchmark_err_type;
                      validation.Y === benchmark_Y;
                      validation.n == benchmark_n;
                      validation.T == benchmark_T;
                      validation.is_stationary == benchmark_is_stationary;
                      validation.model_struct == benchmark_model_struct;
                      validation.model_args == benchmark_model_args;
                      validation.verb == benchmark_verb;
                      validation.verb_estim == benchmark_verb_estim;
                      validation.weights == benchmark_weights;
                      validation.t0 == benchmark_t0;
                      validation.subsample == benchmark_subsample;
                      validation.max_samples == benchmark_max_samples;
                      validation.log_folder_path == benchmark_log_folder_path]);
end

function validation_input_test(validation_a::ValidationSettings, validation_b::ValidationSettings)

    return ~(false in [validation_a.err_type == validation_b.err_type;
                      validation_a.Y === validation_b.Y;
                      validation_a.n == validation_b.n;
                      validation_a.T == validation_b.T;
                      validation_a.is_stationary == validation_b.is_stationary;
                      validation_a.model_struct == validation_b.model_struct;
                      validation_a.model_args == validation_b.model_args;
                      validation_a.verb == validation_b.verb;
                      validation_a.verb_estim == validation_b.verb_estim;
                      validation_a.weights == validation_b.weights;
                      validation_a.t0 == validation_b.t0;
                      validation_a.subsample == validation_b.subsample;
                      validation_a.max_samples == validation_b.max_samples;
                      validation_a.log_folder_path == validation_b.log_folder_path]);
end

"""
    grid_input_test(grid::HyperGrid, benchmark_data::Tuple)

Check whether the inputs in `grid` are equal to those in `benchmark_data`.
"""
function grid_input_test(grid::HyperGrid, benchmark_data::Tuple)

    benchmark_p, benchmark_λ, benchmark_α, benchmark_β, benchmark_draws = benchmark_data;
    
    return ~(false in [grid.p == benchmark_p;
                      grid.λ == benchmark_λ;
                      grid.α == benchmark_α;
                      grid.β == benchmark_β;
                      grid.draws == benchmark_draws]);
end

"""
    validation_settings_tests(model_args::Tuple, weights::FloatVector, t0::Int64, subsample::Float64, max_samples::Int64, log_folder_path::String, benchmark_data::Tuple)

Run basic tests on ValidationSettings.
"""
function validation_settings_tests(model_args::Tuple, weights::FloatVector, t0::Int64, subsample::Float64, max_samples::Int64, log_folder_path::String, benchmark_data::Tuple)

    err_type, Y, n, T, is_stationary, model_struct, _ = benchmark_data;

    # Default optional arguments
    validation_1 = ValidationSettings(err_type, Y, is_stationary, model_struct);
    validation_2 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=());
    validation_3 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true);
    validation_4 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true, verb_estim=false);
    validation_5 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true, verb_estim=false, weights=nothing);
    validation_6 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true, verb_estim=false, weights=nothing, t0=nothing);
    validation_7 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing);
    validation_8 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing);
    validation_9 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=(), verb=true, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_1, benchmark_data);
    @test validation_input_test(validation_1, validation_2);
    @test validation_input_test(validation_1, validation_3);
    @test validation_input_test(validation_1, validation_4);
    @test validation_input_test(validation_1, validation_5);
    @test validation_input_test(validation_1, validation_6);
    @test validation_input_test(validation_1, validation_7);
    @test validation_input_test(validation_1, validation_8);
    @test validation_input_test(validation_1, validation_9);

    # Varying model_args
    validation_10 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args);
    validation_11 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true);
    validation_12 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true, verb_estim=false);
    validation_13 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true, verb_estim=false, weights=nothing);
    validation_14 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true, verb_estim=false, weights=nothing, t0=nothing);
    validation_15 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing);
    validation_16 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing);
    validation_17 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=true, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_10, validation_11);
    @test validation_input_test(validation_10, validation_12);
    @test validation_input_test(validation_10, validation_13);
    @test validation_input_test(validation_10, validation_14);
    @test validation_input_test(validation_10, validation_15);
    @test validation_input_test(validation_10, validation_16);
    @test validation_input_test(validation_10, validation_17);

    # Custom model_args, verb
    validation_18 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false);
    validation_19 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=false);
    validation_20 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=false, weights=nothing);
    validation_21 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=false, weights=nothing, t0=nothing);
    validation_22 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing);
    validation_23 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing);
    validation_24 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=false, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_18, validation_19);
    @test validation_input_test(validation_18, validation_20);
    @test validation_input_test(validation_18, validation_21);
    @test validation_input_test(validation_18, validation_22);
    @test validation_input_test(validation_18, validation_23);
    @test validation_input_test(validation_18, validation_24);

    # Custom model_args, verb, verb_estim
    validation_25 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true);
    validation_26 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=nothing);
    validation_27 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=nothing, t0=nothing);
    validation_28 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=nothing, t0=nothing, subsample=nothing);
    validation_29 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing);
    validation_30 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=nothing, t0=nothing, subsample=nothing, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_25, validation_26);
    @test validation_input_test(validation_25, validation_27);
    @test validation_input_test(validation_25, validation_28);
    @test validation_input_test(validation_25, validation_29);
    @test validation_input_test(validation_25, validation_30);

    # Custom model_args, verb, verb_estim, weights
    validation_31 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights);
    validation_32 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=nothing);
    validation_33 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=nothing, subsample=nothing);
    validation_34 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=nothing, subsample=nothing, max_samples=nothing);
    validation_35 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=nothing, subsample=nothing, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_31, validation_32);
    @test validation_input_test(validation_31, validation_33);
    @test validation_input_test(validation_31, validation_34);
    @test validation_input_test(validation_31, validation_35);

    # Custom model_args, verb, verb_estim, weights, t0
    validation_36 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0);
    validation_37 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=nothing);
    validation_38 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=nothing, max_samples=nothing);
    validation_39 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=nothing, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_36, validation_37);
    @test validation_input_test(validation_36, validation_38);
    @test validation_input_test(validation_36, validation_39);

    # Custom model_args, verb, verb_estim, weights, t0, subsample
    validation_40 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=subsample);
    validation_41 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=subsample, max_samples=nothing);
    validation_42 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=subsample, max_samples=nothing, log_folder_path=nothing);

    @test validation_input_test(validation_40, validation_41);
    @test validation_input_test(validation_40, validation_42);

    # Custom model_args, verb, verb_estim, weights, t0, subsample, max_samples
    validation_43 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples);
    validation_44 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=nothing);

    @test validation_input_test(validation_43, validation_44);

    # Custom model_args, verb, verb_estim, weights, t0, subsample, max_samples
    validation_45 = ValidationSettings(err_type, Y, is_stationary, model_struct, model_args=model_args, verb=false, verb_estim=true, weights=weights, t0=t0, subsample=subsample, max_samples=max_samples, log_folder_path=log_folder_path);

    @test validation_45.log_folder_path == log_folder_path;
end

@testset "ValidationSettings and HyperGrid: basic functionalities" begin
    
    # Initialise data and benchmark data
    Y = [0.72 missing 1.86 missing missing 2.52 2.98 3.81 missing 4.36;
         0.95 0.70 missing missing missing missing 2.84 3.88 3.84 4.63];

    n = size(Y,1);
    T = size(Y,2);

    # Tests on ValidationSettings
    benchmark_data_vs = (4, Y, n, T, true, VARSettings, (), true, false, nothing, nothing, nothing, nothing, nothing);
    validation_settings_tests(("something", 123), collect(1.0:0.5:5.5), 100, 0.25, 1000, "some folder", benchmark_data_vs);

    # Tests on HyperGrid
    grid = HyperGrid([1; 10], [0; 0.5], [0; 1], [0; 10], 500);
    @test grid_input_test(grid, ([1; 10], [0; 0.5], [0; 1], [0; 10], 500));
end

@testset "select_hyperparameters: iis and oos" begin

    # Simulate VAR(q) data with mean zero and unit standard deviation
    rng = StableRNG(1);
    q = 2;
    n = 2;
    T = 200;
    true_coefficient = [0.3 0.0 0.0 0.0; 0.0 -0.1 0.0 0.0];

    v = randn(rng, n, T+q);
    Y = zeros(n, T+q);
    for t=1:T+q
        if t <= q
            Y[:,t] .= 0.0;
        else
            Y[:,t] = true_coefficient*vcat([Y[:,t-lag][:] for lag=1:q]...) + v[:,t];
        end
    end
    Y = standardise(Y[:,q+1:end]);

    # Setup in-sample and pseudo out-of-sample validation
    grid = HyperGrid([4, 4], [1, 5], [0, 1], [1, 5], 10);
    validation_settings_iis = ValidationSettings(1, Y, true, VARSettings, verb=false);
    validation_settings_oos = ValidationSettings(2, Y, true, VARSettings, t0=100, verb=false);

    # Run in-sample validation
    candidates_iis, errors_iis = select_hyperparameters(validation_settings_iis, grid, 2);

    # Run pseudo out-of-sample validation
    candidates_oos, errors_oos = select_hyperparameters(validation_settings_oos, grid, 2);

    # Load benchmarks
    benchmark_candidates = read_test_input("./input/validation/candidates");
    benchmark_errors_iis = read_test_input("./input/validation/errors_iis");
    benchmark_errors_oos = read_test_input("./input/validation/errors_oos");

    # Run tests
    @test norm(candidates_iis-benchmark_candidates, Inf) <= 1e-8;
    @test candidates_iis == candidates_oos;
    @test norm(errors_iis-benchmark_errors_iis, Inf) <= 1e-8;
    @test norm(errors_oos-benchmark_errors_oos, Inf) <= 1e-8;
end