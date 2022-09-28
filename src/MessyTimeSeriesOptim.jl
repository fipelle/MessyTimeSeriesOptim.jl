__precompile__()

module MessyTimeSeriesOptim

	# Libraries
	using Dates, Distributed, Logging, LoopVectorization;
	using Distributions, LinearAlgebra, SparseArrays, StableRNGs, Statistics;
	using MessyTimeSeries, Optimization, OptimizationNLopt;
	using Infiltrator;
	
	# Aliases for MessyTimeSeries
	find_observed_data = MessyTimeSeries.find_observed_data;
	update_smoothing_factors! = MessyTimeSeries.update_smoothing_factors!;
	backwards_pass = MessyTimeSeries.backwards_pass;

	# Custom dependencies
	local_path = dirname(@__FILE__);
	include("$(local_path)/types.jl");
	include("$(local_path)/initialisation.jl");
	include("$(local_path)/estimation.jl");
	include("$(local_path)/validation.jl");
	include("$(local_path)/models/dfm.jl");
	include("$(local_path)/models/var.jl");
	include("$(local_path)/models/vma.jl");

	# Export
	export EstimSettings, DFMSettings, VARSettings, VMASettings, ValidationSettings, HyperGrid;
	export build_Γ;
	export initial_univariate_decomposition_kitagawa, initial_univariate_decomposition_llt;
	export ecm;
	export select_hyperparameters, fc_err, jackknife_err;
end
