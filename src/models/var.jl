#=
--------------------------------------------------------------------------------------------------------------------------------
Types and constructors
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    VARSettings(...)

Define an immutable structure used to initialise the estimation routine for VAR(q) models.

# Arguments
- `Y`: observed measurements (`nxT`)
- `n`: Number of series
- `T`: Number of observations
- `q`: Order of the autoregressive polynomial
- `nq`: n*q
- `m`: n*q
- `λ`: overall shrinkage hyper-parameter for the elastic-net penalty
- `α`: weight associated to the LASSO component of the elastic-net penalty
- `β`: additional shrinkage for distant lags (p>1)
- `Γ`: Diagonal matrix used to input the hyperparameters in the estimation - see Pellegrino (2022) for details
- `ε`: Small number (default: 1e-4)
- `tol`: tolerance used to check convergence (default: 1e-4)
- `max_iter`: maximum number of iterations for the estimation algorithm (default: 1000)
- `prerun`: number of iterations prior the actual estimation algorithm (default: 2)
- `check_quantile`: check the quantile of the relative change of the parameters for convergence purposes (default: false)
- `verb`: Verbose output (default: true)
"""
struct VARSettings <: EstimSettings
    Y::Union{FloatMatrix, JMatrix{Float64}}
    n::Int64
    T::Int64
    q::Int64
    nq::Int64
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

# VARSettings constructor
function VARSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, q::Int64, λ::Float64, α::Float64, β::Float64; ε::Float64=1e-4, tol::Float64=1e-4, max_iter::Int64=1000, prerun::Int64=2, check_quantile::Bool=false, verb::Bool=true)

    # Compute missing inputs
    n, T = size(Y);
    m = n*q;
    Γ = build_Γ(n, q, λ, β);

    # Return VARSettings
    return VARSettings(Y, n, T, q, n*q, m, λ, α, β, Γ, ε, tol, max_iter, prerun, check_quantile, verb);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Initialisation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    check_model_bounds(estim::VARSettings)

Check whether the arguments for the estimation of a VAR(q) model are correct.
"""
function check_model_bounds(estim::VARSettings)
    check_bounds(estim.q, 0);
    check_bounds(estim.λ, 0);
    check_bounds(estim.α, 0, 1);
    check_bounds(estim.β, 1);
    check_bounds(estim.max_iter, 3);
    check_bounds(estim.max_iter, estim.prerun);
    check_bounds(estim.n, 2); # It supports only multivariate models (for now ...)
end

"""
    initialise(estim::VARSettings)

Initialise the ECM algorithm for VAR(q) models.

# Arguments
- `estim`: settings used for the estimation

# References
Pellegrino (2022)
"""
function initialise(estim::VARSettings)
    
    # Interpolate data
    data = interpolate_series(estim.Y, estim.n, estim.T);

    # VAR(q) data
    Y, Y_lagged = lag(data, estim.q);
    
    # Estimate ridge VAR(q) coefficients
    C_star_init = Y*Y_lagged'/Symmetric(Y_lagged*Y_lagged' + estim.Γ);
    enforce_causality_and_invertibility!(C_star_init);

    # Estimate var-cov matrix of the residuals
    U = Y - C_star_init*Y_lagged;
    Q_init = Symmetric((U*U')/(estim.T - estim.q))::SymMatrix;

    # Initialise B_star
    B_star_init = zeros(estim.n, estim.m-estim.n);

    sspace = KalmanSettings(estim.Y, sspace_representation(estim, B_star_init, C_star_init, Q_init)..., compute_loglik=false);
    
    # Coordinates
    coordinates_transition_current = collect(1:estim.n);
    coordinates_transition_lagged = collect(1:estim.m);
    coordinates_transition_PPs = coordinates_transition_lagged .+ estim.n;
    coordinates_transition_P0 = LinearIndices(sspace.P0)[:];
    coordinates_free_params_C = CartesianIndices(sspace.C)[1:estim.n, 1:estim.m][:];

    # Useful views
    C_star = @view sspace.C[1:estim.n, 1:estim.m]; # in the case of the VAR this is equivalent to the cm view
    Q_view = @view sspace.DQD.data[1:sspace.Y.n, 1:sspace.Y.n];

    # Return output
    return sspace, nothing, C_star, Q_view, nothing, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs, coordinates_transition_P0, nothing, coordinates_free_params_C;
end

#=
--------------------------------------------------------------------------------------------------------------------------------
CM-step
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    cm_step!(estim::VARSettings, sspace::KalmanSettings, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, F_raw::FloatMatrix, G::FloatMatrix, H_raw::FloatMatrix, M::Nothing, N::Nothing, O::Nothing, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

Run the CM-step for elastic-net VAR(q) models.
"""
function cm_step!(estim::VARSettings, sspace::KalmanSettings, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, F_raw::FloatMatrix, G::FloatMatrix, H_raw::FloatMatrix, M::Nothing, N::Nothing, O::Nothing, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

    # Make sure `F_raw` and `H_raw` are symmetric
    F = Symmetric(F_raw)::SymMatrix;
    H = Symmetric(H_raw)::SymMatrix;

    # Companion forms
    C_companion_old = sspace.C[1:estim.m, 1:estim.m];
    C_companion_new = @view sspace.C[1:estim.m, 1:estim.m];

    # Compute inverse of Q
    inv_Q = inv(Symmetric(Q_view));

    # Loop over the free parameters
    for ij in coordinates_free_params_C

        # Coordinates
        i,j = ij.I;

        # Compute new value for sspace.C[ij]
        C_star[ij] = soft_thresholding(turbo_dot(inv_Q[:,i], G[:,j] - C_star*H[:,j]) + inv_Q[i,i]*C_star[ij]*H[j,j], 0.5*estim.α*estim.Γ[j,j]);
        C_star[ij] /= inv_Q[i,i]*H[j,j] + (1-estim.α)*estim.Γ[j,j];
    end

    # Adjust coefficients (if needed)
    enforce_causality_and_invertibility!(C_companion_new, C_companion_old);

    # CM-step for Q
    Q_cm_step = (F-G*C_star'-C_star*G'+C_star*H*C_star')/estim.T;
    for ij in eachindex(sspace.Q) # computing eachindex on sspace.Q is convenient since it gives Cartesian indices aligned with sspace.DQD
        Q_view[ij] = Q_cm_step[ij];
    end
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Model validation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    VARSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::Nothing, p::Int64, λ::Float64, α::Float64, β::Float64)
    VARSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::NamedTuple, p::Int64, λ::Float64, α::Float64, β::Float64)

Generate the relevant VARSettings structure by using the candidate hyperparameters `p`, `λ`, `α`, `β`.
"""
VARSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::Nothing, p::Int64, λ::Float64, α::Float64, β::Float64) = VARSettings(Y, p, λ, α, β, verb=false);
VARSettings(Y::Union{FloatMatrix, JMatrix{Float64}}, model_args::Tuple, model_kwargs::NamedTuple, p::Int64, λ::Float64, α::Float64, β::Float64) = VARSettings(Y, p, λ, α, β; model_kwargs...);