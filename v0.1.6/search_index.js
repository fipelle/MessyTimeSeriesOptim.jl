var documenterSearchIndex = {"docs":
[{"location":"man/estimation/#Estimation","page":"Estimation","title":"Estimation","text":"","category":"section"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"MessyTimeSeriesOptim provides a generic ECM algorithm easily adaptable to any model that can be written in a linear state-space form. This algorithm is also compatible with penalised maximum likelihood and penalised quasi maximum likelihood estimation.","category":"page"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"Pages = [\"estimation.md\"]\nDepth = 2","category":"page"},{"location":"man/estimation/#ECM-algorithm","page":"Estimation","title":"ECM algorithm","text":"","category":"section"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"ecm","category":"page"},{"location":"man/estimation/#MessyTimeSeriesOptim.ecm","page":"Estimation","title":"MessyTimeSeriesOptim.ecm","text":"ecm(estim::EstimSettings)\n\nRun the ECM algorithm in Pellegrino (2022).\n\nArguments\n\nestim: settings used for the estimation\noutput_sspace_data: Optional argument. If specified, it is used as the output state space data. Otherwise, estim.Y is used instead.\n\nReferences\n\nPellegrino (2022)\n\n\n\n\n\n","category":"function"},{"location":"man/estimation/#EstimSettings","page":"Estimation","title":"EstimSettings","text":"","category":"section"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"MessyTimeSeriesOptim provides the following EstimSettings subtypes.","category":"page"},{"location":"man/estimation/#Dynamic-factor-models","page":"Estimation","title":"Dynamic factor models","text":"","category":"section"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"DFMSettings","category":"page"},{"location":"man/estimation/#MessyTimeSeriesOptim.DFMSettings","page":"Estimation","title":"MessyTimeSeriesOptim.DFMSettings","text":"DFMSettings(...)\n\nDefine an immutable structure used to initialise the estimation routine for a DFM.\n\nArguments\n\nY: observed measurements (nxT)\nn: Number of series\nT: Number of observations\nlags: Order of the autoregressive polynomial of each common cycle\nn_trends: Number of trends\nn_drifts: Number of drifts\nn_cycles: Number of cycles\nn_non_stationary: ntrends + ndrifts\nm: nnonstationary + n_cycles*lags + n\ntrends_skeleton: The basic structure for the trends loadings (or nothing)\ncycles_skeleton: The basic structure for the cycles loadings (or nothing)\ndrifts_selection: BitArray{1} identifying which trend has a drift to estimate (or nothing)\ntrends_free_params: BitArray{2} identifying the trend loadings to estimate (or nothing)\ncycles_free_params: BitArray{2} identifying the cycle loadings to estimate (or nothing)\nλ: overall shrinkage hyper-parameter for the elastic-net penalty\nα: weight associated to the LASSO component of the elastic-net penalty\nβ: additional shrinkage for distant lags (p>1)\nΓ: Diagonal matrix used to input the hyperparameters in the penalty computation for the common cycles\nΓ_idio: Diagonal matrix used to input the hyperparameters in the penalty computation for the idiosyncratic cycles\nΓ_extended: Diagonal matrix used to input the hyperparameters in the estimation\nε: Small number (default: 1e-4)\ntol: tolerance used to check convergence (default: 1e-4)\nmax_iter: maximum number of iterations for the estimation algorithm (default: 1000)\nprerun: number of iterations prior the actual estimation algorithm (default: 2)\ncheck_quantile: check the quantile of the relative change of the parameters for convergence purposes (default: false)\nverb: Verbose output (default: true)\n\n\n\n\n\n","category":"type"},{"location":"man/estimation/#Vector-autoregressions","page":"Estimation","title":"Vector autoregressions","text":"","category":"section"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"VARSettings","category":"page"},{"location":"man/estimation/#MessyTimeSeriesOptim.VARSettings","page":"Estimation","title":"MessyTimeSeriesOptim.VARSettings","text":"VARSettings(...)\n\nDefine an immutable structure used to initialise the estimation routine for VAR(q) models.\n\nArguments\n\nY: observed measurements (nxT)\nn: Number of series\nT: Number of observations\nq: Order of the autoregressive polynomial\nnq: n*q\nm: n*q\nλ: overall shrinkage hyper-parameter for the elastic-net penalty\nα: weight associated to the LASSO component of the elastic-net penalty\nβ: additional shrinkage for distant lags (p>1)\nΓ: Diagonal matrix used to input the hyperparameters in the estimation - see Pellegrino (2022) for details\nε: Small number (default: 1e-4)\ntol: tolerance used to check convergence (default: 1e-4)\nmax_iter: maximum number of iterations for the estimation algorithm (default: 1000)\nprerun: number of iterations prior the actual estimation algorithm (default: 2)\ncheck_quantile: check the quantile of the relative change of the parameters for convergence purposes (default: false)\nverb: Verbose output (default: true)\n\n\n\n\n\n","category":"type"},{"location":"man/estimation/#Vector-moving-averages","page":"Estimation","title":"Vector moving averages","text":"","category":"section"},{"location":"man/estimation/","page":"Estimation","title":"Estimation","text":"VMASettings","category":"page"},{"location":"man/estimation/#MessyTimeSeriesOptim.VMASettings","page":"Estimation","title":"MessyTimeSeriesOptim.VMASettings","text":"VMASettings(...)\n\nDefine an immutable structure used to initialise the estimation routine for VMA(r) models.\n\nArguments\n\nY: observed measurements (nxT)\nn: Number of series\nT: Number of observations\nr: Order of the moving average polynomial\nnr: n*r\nm: nr+n*1_{q=0}\nλ: overall shrinkage hyper-parameter for the elastic-net penalty\nα: weight associated to the LASSO component of the elastic-net penalty\nβ: additional shrinkage for distant lags (p>1)\nΓ: Diagonal matrix used to input the hyperparameters in the estimation - see Pellegrino (2022) for details\nε: Small number (default: 1e-4)\ntol: tolerance used to check convergence (default: 1e-4)\nmax_iter: maximum number of iterations for the estimation algorithm (default: 1000)\nprerun: number of iterations prior the actual estimation algorithm (default: 2)\ncheck_quantile: check the quantile of the relative change of the parameters for convergence purposes (default: false)\nverb: Verbose output (default: true)\n\n\n\n\n\n","category":"type"},{"location":"man/validation/#Validation","page":"Validation","title":"Validation","text":"","category":"section"},{"location":"man/validation/","page":"Validation","title":"Validation","text":"MessyTimeSeriesOptim has a generic implementation to validate the models in @ref(estimation) when needed. This is done via the function select_hyperparameters.","category":"page"},{"location":"man/validation/","page":"Validation","title":"Validation","text":"Pages = [\"validation.md\"]\nDepth = 2","category":"page"},{"location":"man/validation/#Functions","page":"Validation","title":"Functions","text":"","category":"section"},{"location":"man/validation/","page":"Validation","title":"Validation","text":"select_hyperparameters","category":"page"},{"location":"man/validation/#MessyTimeSeriesOptim.select_hyperparameters","page":"Validation","title":"MessyTimeSeriesOptim.select_hyperparameters","text":"select_hyperparameters(validation_settings::ValidationSettings, γ_grid::HyperGrid)\n\nSelect the tuning hyper-parameters for the elastic-net vector autoregression.\n\nArguments\n\nvalidation_settings: ValidationSettings struct\nγ_grid: HyperGrid struct\n\nReferences\n\nPellegrino (2022)\n\n\n\n\n\n","category":"function"},{"location":"man/validation/#Types","page":"Validation","title":"Types","text":"","category":"section"},{"location":"man/validation/","page":"Validation","title":"Validation","text":"ValidationSettings","category":"page"},{"location":"man/validation/#MessyTimeSeriesOptim.ValidationSettings","page":"Validation","title":"MessyTimeSeriesOptim.ValidationSettings","text":"ValidationSettings(...)\n\nDefine an immutable structure used to define the validation settings.\n\nThe arguments are two dimensional arrays representing the bounds of the grid for each hyperparameter.\n\nArguments\n\nerr_type:\n1 In-sample error\n2 Out-of-sample error\n3 Block jackknife error\n4 Artificial jackknife error\nY: observed measurements (nxT)\nn: Number of series\nT: Number of observations\nis_stationary: Boolean value\nmodel_struct: DataType identifying the estimation structure to use\nmodel_args: Tuple with the arguments required to setup the model specified in model_struct (irrelevant for VARs and VMAs)\nmodel_kwargs: Tuple with the keyword arguments required to setup the model specified in model_struct (default: nothing)\ncoordinates_params_rescaling: Array of vectors including information on the parameters (if any) that require to be rescaled to match the data standardisation (default: nothing)\nverb: Verbose output (default: true)\nverb_estim: Further verbose output (default: false)\nweights: Weights for the forecast error. standardise_error has priority over weights. (default: ones(n))\nt0: weight associated to the LASSO component of the elastic-net penalty\nsubsample: number of observations removed in the subsampling process, as a percentage of the original sample size. It is bounded between 0 and 1.\nmax_samples: if C(n*T,d) is large, artificialjackknife would generate `maxsamples` jackknife samples. (used only for the artificial jackknife)\nlog_folder_path: folder to store the log file. When this file is defined then the stdout is redirected to this file.\n\n\n\n\n\n","category":"type"},{"location":"man/validation/","page":"Validation","title":"Validation","text":"HyperGrid","category":"page"},{"location":"man/validation/#MessyTimeSeriesOptim.HyperGrid","page":"Validation","title":"MessyTimeSeriesOptim.HyperGrid","text":"HyperGrid(...)\n\nDefine an immutable structure used to define the grid of hyperparameters used in validation(...).\n\nThe arguments are two dimensional arrays representing the bounds of the grid for each hyperparameter.\n\nArguments\n\np: Number of lags\nλ: overall shrinkage hyper-parameter for the elastic-net penalty\nα: weight associated to the LASSO component of the elastic-net penalty\nβ: additional shrinkage for distant lags (p>1)\ndraws: number of draws used to construct the grid of candidates\n\n\n\n\n\n","category":"type"},{"location":"#MessyTimeSeriesOptim.jl","page":"MessyTimeSeriesOptim.jl","title":"MessyTimeSeriesOptim.jl","text":"","category":"section"},{"location":"","page":"MessyTimeSeriesOptim.jl","title":"MessyTimeSeriesOptim.jl","text":"MessyTimeSeriesOptim includes estimation and validation algorithms for time series models, compatible with incomplete data.","category":"page"},{"location":"#Outline","page":"MessyTimeSeriesOptim.jl","title":"Outline","text":"","category":"section"},{"location":"","page":"MessyTimeSeriesOptim.jl","title":"MessyTimeSeriesOptim.jl","text":"Pages = [\n    \"man/estimation.md\",\n    \"man/validation.md\",\n]\nDepth = 3","category":"page"}]
}
