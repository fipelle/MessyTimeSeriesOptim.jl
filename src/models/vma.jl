#=
--------------------------------------------------------------------------------------------------------------------------------
Types and constructors
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    VMASettings(...)

Define an immutable structure used to initialise the estimation routine for VMA(r) models.

# Arguments
- `Y`: observed measurements (`nxT`)
- `n`: Number of series
- `T`: Number of observations
- `r`: Order of the moving average polynomial
- `nr`: n*r
- `m`: nr+n*1_{q=0}
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `Γ`: Diagonal matrix used to input the hyperparameters in the estimation - see Pellegrino (2021) for details
- `ε`: Small number (default: 1e-4)
- `tol`: tolerance used to check convergence (default: 1e-4)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual estimation algorithm (default: 2)
- `check_quantile`: check the quantile of the relative change of the parameters for convergence purposes (default: false)
- `verb`: Verbose output (default: true)
"""
struct VMASettings <: EstimSettings
    Y::Union{FloatMatrix, JMatrix{Float64}}
    n::Int64
    T::Int64
    r::Int64
    nr::Int64
    m::Int64
    λ::Float64
    α::Float64
    β::Float64
    Γ::DiagMatrix
    ε::Float64
    tol::Float64
    max_iter::Int64
    prerun::Int64
    check_quantile::Bool
    verb::Bool
end

# VMASettings constructor
function VMASettings(Y::Union{FloatMatrix, JMatrix{Float64}}, r::Int64, λ::Float64, α::Float64, β::Float64; ε::Float64=1e-4, tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, check_quantile::Bool=false, verb::Bool=true)

    # Compute missing inputs
    n, T = size(Y);
    m = n*r+n;
    Γ = build_Γ(n, r, λ, β);

    # Return VMASettings
    return VMASettings(Y, n, T, r, n*r, m, λ, α, β, Γ, ε, tol, max_iter, prerun, check_quantile, verb);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Initialisation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    check_model_bounds(estim::VMASettings)

Check whether the arguments for the estimation of a VMA(r) model are correct.
"""
function check_model_bounds(estim::VMASettings)
    check_bounds(estim.r, 0);
    check_bounds(estim.λ, 0);
    check_bounds(estim.α, 0, 1);
    check_bounds(estim.β, 1);
    check_bounds(estim.max_iter, 3);
    check_bounds(estim.max_iter, estim.prerun);
    check_bounds(estim.n, 2); # It supports only multivariate models (for now ...)
end

"""
    initialise(estim::VMASettings)

Initialise the ECM algorithm for VMA(r) models.

# Arguments
- `estim`: settings used for the estimation

# References
Pellegrino (2021)
"""
function initialise(estim::VMASettings)

    # Initialise C_star
    C_star_init = zeros(estim.n, estim.m)

    # Interpolate data
    data = interpolate_series(estim.Y, estim.n, estim.T);

    # VAR(sqrt(T)) data
    sqrt_T = floor(sqrt(estim.T)) |> Int64;
    Y, Y_lagged = lag(data, sqrt_T);

    # Corresponding coefficients
    Γ_sqrt_T = build_Γ(estim.n, sqrt_T, estim.λ, estim.β);
    VAR_coefficients = Y*Y_lagged'/Symmetric(Y_lagged*Y_lagged' + Γ_sqrt_T);
    enforce_causality_and_invertibility!(VAR_coefficients);

    # Residuals
    residuals = Y - VAR_coefficients*Y_lagged;

    # VMA(r) data
    Y_VMA, _ = lag(Y, estim.r);
    U, U_VMA = lag(residuals, estim.r);

    # Initialise VMA
    B_star_init = Y_VMA*U_VMA'/Symmetric(U_VMA*U_VMA' + estim.Γ);
    enforce_causality_and_invertibility!(B_star_init);
    Q_init = Symmetric((U*U')/size(U,2));

    sspace = KalmanSettings(estim.Y, sspace_representation(estim, B_star_init, C_star_init, Q_init)..., compute_loglik=false);

    # Coordinates
    coordinates_measurement_states = collect(1:estim.m);
    coordinates_transition_current = collect(1:estim.n);
    coordinates_transition_P0 = LinearIndices(sspace.P0)[:];
    coordinates_free_params_B = CartesianIndices(sspace.B)[:, 1+estim.n:estim.m][:];

    # Useful views
    B_star = @view sspace.B[:, coordinates_measurement_states];
    Q_view = @view sspace.DQD.data[1:sspace.Y.n, 1:sspace.Y.n];

    # Return output
    return sspace, B_star, nothing, Q_view, coordinates_measurement_states, coordinates_transition_current, nothing, nothing, coordinates_transition_P0, coordinates_free_params_B, nothing;
end

#=
--------------------------------------------------------------------------------------------------------------------------------
CM-step
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    cm_step!(estim::VMASettings, sspace::KalmanSettings, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, F_raw::FloatMatrix, G::Nothing, H::Nothing, M::FloatArray, N::Array{VectorsArray{Float64},1}, O::Array{VectorsArray{Float64},1}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing)

Run the CM-step for elastic-net VMA(r) models.
"""
function cm_step!(estim::VMASettings, sspace::KalmanSettings, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, F_raw::FloatMatrix, G::Nothing, H::Nothing, M::FloatArray, N::Array{VectorsArray{Float64},1}, O::Array{VectorsArray{Float64},1}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing)

    # Make sure F is symmetric
    F = Symmetric(F_raw)::SymMatrix;

    # Companion forms: initial settings
    B_companion_params = @view sspace.B[:, 1+estim.n:estim.m];
    B_companion_old = companion_form(B_companion_params, extended=false);

    # Loop over the free parameters
    for ij in coordinates_free_params_B

        # Compute remaining sufficient statistics
        sum_numerator, sum_denominator, i, j = cm_step_time_loop(sspace, B_star, ij, N, O);

        # Compute new value for sspace.B[ij]
        B_star[ij] = soft_thresholding(M[ij] - sum_numerator, 0.5*estim.α*estim.ε*estim.Γ[j-estim.n, j-estim.n]);
        B_star[ij] /= sum_denominator + (1-estim.α)*estim.ε*estim.Γ[j-estim.n, j-estim.n];
    end

    # Companion forms: final settings
    B_companion_new = companion_form(B_companion_params, extended=false);

    # Adjust coefficients (if needed)
    enforce_causality_and_invertibility!(B_companion_new, B_companion_old, B_companion_params);

    # CM-step for Q
    for ij in eachindex(F)
        Q_view[ij] = F[ij]/estim.T;
    end
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Model validation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    VMASettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::Nothing, p::Int64, λ::Float64, α::Float64, β::Float64)
    VMASettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::NamedTuple, p::Int64, λ::Float64, α::Float64, β::Float64)

Generate the relevant VMASettings structure by using the candidate hyperparameters `p`, `λ`, `α`, `β`.
"""
VMASettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::Nothing, p::Int64, λ::Float64, α::Float64, β::Float64) = VMASettings(Y, p, λ, α, β, verb=false);
VMASettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::NamedTuple, p::Int64, λ::Float64, α::Float64, β::Float64) = VMASettings(Y, p, λ, α, β; model_kwargs...);