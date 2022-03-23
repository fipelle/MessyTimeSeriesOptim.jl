# Aliases (types)
const BitVector        = BitArray{1};
const BitMatrix        = BitArray{2};
const CoordinatesVector = Array{CartesianIndex{2},1};
const VectorsArray{T}   = Array{Vector{T},1}; # TBD: change to ArrayOfVectors

# Estimation structures
abstract type EstimSettings end

# Constructor for Γ
function build_vec_Γ(n::Int64, p::Int64, β::Float64)

    vec_Γ = Array{Float64,1}();
    for i=0:p-1
        for j=1:n
            push!(vec_Γ, β^i);
        end
    end

    return vec_Γ;
end

function build_Γ(n::Int64, p::Int64, λ::Float64, β::Float64)
    return Diagonal(λ.*build_vec_Γ(n, p, β))::DiagMatrix;
end

"""
    SmootherArrays(...)

Internal re-usable arrays for the Kalman smoother output.
"""
struct SmootherArrays
    J1::FloatVector
    J2::SymMatrix
    Xs_leading::FloatVector
    Ps_leading::SymMatrix
    F::Union{FloatMatrix, Nothing}
    G::Union{FloatMatrix, Nothing}
    H::Union{FloatMatrix, Nothing}
    M::Union{FloatMatrix, Nothing}
    N::Union{Vector{SparseMatrixCSC{Float64, Int64}}, Nothing}
    O::Union{Array{VectorsArray{Float64},1}, Nothing}
    buffer_M::Union{FloatVector, Nothing}
    buffer_O::Union{SymMatrix, Nothing}
end

# Constructor for SmootherArrays
function SmootherArrays(estim::EstimSettings, sspace::KalmanSettings, coordinates_measurement_states::Union{IntVector, Nothing}, coordinates_transition_current::Union{IntVector, Nothing}, coordinates_transition_lagged::Union{IntVector, Nothing}, coordinates_transition_PPs::Union{IntVector, Nothing})
    
    # Kalman smoother internal arrays
    J1 = zeros(sspace.m);
    J2 = Symmetric(zeros(sspace.m, sspace.m));
    Xs_leading = zeros(sspace.m);
    Ps_leading = Symmetric(zeros(sspace.m, sspace.m));

    # ECM statistics for the transition equation
    F, G, H = initialise_ecm_stats_transition(coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);

    # ECM statistics for the measurement equation
    M, N, O, buffer_M, buffer_O = initialise_ecm_stats_measurement(estim, coordinates_measurement_states);

    # Return structure
    return SmootherArrays(J1, J2, Xs_leading, Ps_leading, F, G, H, M, N, O, buffer_M, buffer_O);
end

# Validation types

"""
    ValidationSettings(...)

Define an immutable structure used to define the validation settings.

The arguments are two dimensional arrays representing the bounds of the grid for each hyperparameter.

# Arguments
- `err_type`:
    - 1 In-sample error
    - 2 Out-of-sample error
    - 3 Block jackknife error
    - 4 Artificial jackknife error
- `Y`: observed measurements (`nxT`)
- `n`: Number of series
- `T`: Number of observations
- `is_stationary`: Boolean value
- `model_struct`: DataType identifying the estimation structure to use
- `model_args`: Tuple with the arguments required to setup the model specified in `model_struct` (irrelevant for VARs and VMAs)
- `model_kwargs`: Tuple with the keyword arguments required to setup the model specified in `model_struct` (default: nothing)
- `coordinates_params_rescaling`: Array of vectors including information on the parameters (if any) that require to be rescaled to match the data standardisation (default: nothing)
- `verb`: Verbose output (default: true)
- `verb_estim`: Further verbose output (default: false)
- `weights`: Weights for the forecast error. standardise_error has priority over weights. (default: ones(n))
- `t0`: weight associated to the LASSO component of the elastic-net penalty
- `subsample`: number of observations removed in the subsampling process, as a percentage of the original sample size. It is bounded between 0 and 1.
- `max_samples`: if `C(n*T,d)` is large, artificial_jackknife would generate `max_samples` jackknife samples. (used only for the artificial jackknife)
- `log_folder_path`: folder to store the log file. When this file is defined then the stdout is redirected to this file.
"""
struct ValidationSettings
    err_type::Int64
    Y::Union{FloatMatrix, JMatrix{Float64}}
    n::Int64
    T::Int64
    is_stationary::Bool
    model_struct::DataType
    model_args::Tuple
    model_kwargs::Union{NamedTuple, Nothing}
    coordinates_params_rescaling::Union{VectorsArray{Int64}, Nothing}
    verb::Bool
    verb_estim::Bool
    weights::Union{FloatVector, Nothing}
    t0::Union{Int64, Nothing}
    subsample::Union{Float64, Nothing}
    max_samples::Union{Int64, Nothing}
    log_folder_path::Union{String, Nothing}
end

# Constructor for ValidationSettings
ValidationSettings(err_type::Int64, Y::Union{FloatMatrix, JMatrix{Float64}}, is_stationary::Bool, model_struct::DataType; model_args::Tuple=(), model_kwargs::Union{NamedTuple, Nothing}=nothing, coordinates_params_rescaling::Union{VectorsArray{Int64}, Nothing}=nothing, verb::Bool=true, verb_estim::Bool=false, weights::Union{FloatVector, Nothing}=nothing, t0::Union{Int64, Nothing}=nothing, subsample::Union{Float64, Nothing}=nothing, max_samples::Union{Int64, Nothing}=nothing, log_folder_path::Union{String, Nothing}=nothing) =
    ValidationSettings(err_type, Y, size(Y,1), size(Y,2), is_stationary, model_struct, model_args, model_kwargs, coordinates_params_rescaling, verb, verb_estim, weights, t0, subsample, max_samples, log_folder_path);

"""
    HyperGrid(...)

Define an immutable structure used to define the grid of hyperparameters used in validation(...).

The arguments are two dimensional arrays representing the bounds of the grid for each hyperparameter.

# Arguments
- `p`: Number of lags
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `draws`: number of draws used to construct the grid of candidates
"""
struct HyperGrid
    p::IntVector
    λ::FloatVector
    α::FloatVector
    β::FloatVector
    draws::Int64
end
