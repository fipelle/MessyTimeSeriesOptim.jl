__precompile__()

module MessyTimeSeriesOptim

	# Libraries
	using Distributed, Dates, Logging, LoopVectorization;
	using LinearAlgebra, Distributions, StableRNGs, Statistics;
	using TSAnalysis;

	# Aliases for TSAnalysis
	find_observed_data = TSAnalysis.find_observed_data;
	update_smoothing_factors! = TSAnalysis.update_smoothing_factors!;
	backwards_pass = TSAnalysis.backwards_pass;

	# Custom dependencies
	const local_path = dirname(@__FILE__);
	include("$local_path/types.jl");
	include("$local_path/models/dfm.jl");
	include("$local_path/models/var.jl");
	include("$local_path/models/vma.jl");
	include("$local_path/estimation.jl");
	include("$local_path/validation.jl");

	# Export
	export EstimSettings, DFMSettings, VARSettings, VMASettings, ValidationSettings, HyperGrid;
	export build_Γ;
	export ecm;
	export select_hyperparameters, fc_err, jackknife_err;
end
