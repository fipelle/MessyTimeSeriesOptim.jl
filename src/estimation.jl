#=
--------------------------------------------------------------------------------------------------------------------------------
Convergence
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    vec_ecm_params(estim::EstimSettings, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing) 

Return the vec form of the free parameters in B_star and Q_view.

    vec_ecm_params(estim::EstimSettings, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

Return the vec form of the free parameters in C_star and Q_view.

    vec_ecm_params(estim::EstimSettings, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

Return the vec form of the free parameters in B_star, C_star and Q_view.
"""
function vec_ecm_params(estim::EstimSettings, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing) 

    # Setup and memory pre-allocation
    counter = 1;
    output = zeros(length(coordinates_free_params_B) + length(Q_view));

    # Add parameters in B_star to output
    for ij in coordinates_free_params_B
        output[counter] = B_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for ij in eachindex(Q_view)
        output[counter] = Q_view[ij];
        counter += 1;
    end

    # Return output
    return output;
end

function vec_ecm_params(estim::EstimSettings, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

    # Setup and memory pre-allocation
    counter = 1;
    output = zeros(length(coordinates_free_params_C) + length(Q_view));

    # Add parameters in C_star to output
    for ij in coordinates_free_params_C
        output[counter] = C_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for ij in eachindex(Q_view)
        output[counter] = Q_view[ij];
        counter += 1;
    end

    # Return output
    return output;
end

function vec_ecm_params(estim::EstimSettings, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

    # Setup and memory pre-allocation
    counter = 1;
    output = zeros(length(coordinates_free_params_B) + length(coordinates_free_params_C) + length(Q_view));

    # Add parameters in B_star to output
    for ij in coordinates_free_params_B
        output[counter] = B_star[ij];
        counter += 1;
    end

    # Add parameters in C_star to output
    for ij in coordinates_free_params_C
        output[counter] = C_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for ij in eachindex(Q_view)
        output[counter] = Q_view[ij];
        counter += 1;
    end

    # Return output
    return output;
end

"""
    vec_ecm_params!(estim::EstimSettings, output::FloatVector, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing)

Update output in-place with the vec form of the free parameters in B_star and Q_view.

    vec_ecm_params!(estim::EstimSettings, output::FloatVector, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

Update output in-place with the vec form of the free parameters in C_star and Q_view.

    vec_ecm_params!(estim::EstimSettings, output::FloatVector, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

Update output in-place with the vec form of the free parameters in B_star, C_star and Q_view.
"""
function vec_ecm_params!(estim::EstimSettings, output::FloatVector, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing) 

    # Setup and memory pre-allocation
    counter = 1;

    # Add parameters in B_star to output
    for ij in coordinates_free_params_B
        output[counter] = B_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for ij in eachindex(Q_view)
        output[counter] = Q_view[ij];
        counter += 1;
    end
end

function vec_ecm_params!(estim::EstimSettings, output::FloatVector, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

    # Setup and memory pre-allocation
    counter = 1;

    # Add parameters in C_star to output
    for ij in coordinates_free_params_C
        output[counter] = C_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for ij in eachindex(Q_view)
        output[counter] = Q_view[ij];
        counter += 1;
    end
end

function vec_ecm_params!(estim::EstimSettings, output::FloatVector, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)

    # Setup and memory pre-allocation
    counter = 1;

    # Add parameters in B_star to output
    for ij in coordinates_free_params_B
        output[counter] = B_star[ij];
        counter += 1;
    end

    # Add parameters in C_star to output
    for ij in coordinates_free_params_C
        output[counter] = C_star[ij];
        counter += 1;
    end

    # Add parameters in Q_view to output
    for ij in eachindex(Q_view)
        output[counter] = Q_view[ij];
        counter += 1;
    end
end

"""
    update_relative_change!(estim::EstimSettings, params_rel_change::FloatVector, ecm_params_new::FloatVector, ecm_params_old::FloatVector)

Update `params_rel_change` in-place.
"""
@inline function update_relative_change!(estim::EstimSettings, params_rel_change::FloatVector, ecm_params_new::FloatVector, ecm_params_old::FloatVector)
    @turbo for i in eachindex(params_rel_change)
        params_rel_change[i] = abs(ecm_params_new[i] - ecm_params_old[i])/(abs(ecm_params_old[i]) + estim.ε);
    end
end

"""
    ecm_converged_median(estim::EstimSettings, params_rel_change::FloatVector, iter::Int64)

Check whether the ecm has converged looking at the median of the parameters relative change.
"""
function ecm_converged_median(estim::EstimSettings, params_rel_change::FloatVector, iter::Int64)

    # Median relative change
    median_params_rel_change = median(params_rel_change);
    if iter-estim.prerun-1 < 10
        verb_message(estim.verb, "ecm > iter=0$(iter-estim.prerun-1), relative parameter convergence criterion=$(round(median_params_rel_change, digits=16))");
    else
        verb_message(estim.verb, "ecm > iter=$(iter-estim.prerun-1), relative parameter convergence criterion=$(round(median_params_rel_change, digits=16))");
    end

    # Check convergence
    convergence_criterion = median_params_rel_change <= estim.tol;
    return convergence_criterion;
end

"""
    ecm_converged_complete(estim::EstimSettings, params_rel_change::FloatVector, iter::Int64)

Check whether the ecm has converged looking at both the median and 95th quantile of the parameters relative change.
"""
function ecm_converged_complete(estim::EstimSettings, params_rel_change::FloatVector, iter::Int64)

    # Median relative change
    median_params_rel_change = median(params_rel_change);
    quantile_params_rel_change = quantile(params_rel_change, 0.95);
    if iter-estim.prerun-1 < 10
        verb_message(estim.verb, "ecm > iter=0$(iter-estim.prerun-1), median convergence criterion=$(round(median_params_rel_change, digits=16)), 95th quantile convergence criterion=$(round(quantile_params_rel_change, digits=16))");
    else
        verb_message(estim.verb, "ecm > iter=$(iter-estim.prerun-1), median convergence criterion=$(round(median_params_rel_change, digits=16)), 95th quantile convergence criterion=$(round(quantile_params_rel_change, digits=16))");
    end

    # Check convergence
    convergence_criterion = (median_params_rel_change <= estim.tol) & (quantile_params_rel_change <= 10*estim.tol)
    return convergence_criterion;
end

"""
    ecm_converged(estim::EstimSettings, params_rel_change::FloatVector, iter::Int64)

Check whether the ecm has converged.
"""
function ecm_converged(estim::EstimSettings, params_rel_change::FloatVector, iter::Int64)
    if estim.check_quantile == false
        return ecm_converged_median(estim, params_rel_change, iter)::Bool;
    else
        return ecm_converged_complete(estim, params_rel_change, iter)::Bool;
    end
end

#=
--------------------------------------------------------------------------------------------------------------------------------
ECM stats
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    initialise_ecm_stats_measurement(estim::EstimSettings, coordinates_measurement_states::IntVector)

Initialise M, N and O.

    initialise_ecm_stats_measurement(estim::EstimSettings, coordinates_measurement_states::Nothing)

Initialise M, N and O to nothing.
"""
function initialise_ecm_stats_measurement(estim::EstimSettings, coordinates_measurement_states::IntVector)
    M = zeros(estim.n, estim.m);
    N = Array{VectorsArray{Float64},1}(undef, estim.T);
    O = Array{VectorsArray{Float64},1}(undef, estim.T);
    buffer_M = zeros(estim.n);
    buffer_O = Symmetric(zeros(estim.m, estim.m));
    return M, N, O, buffer_M, buffer_O;
end

initialise_ecm_stats_measurement(estim::EstimSettings, coordinates_measurement_states::Nothing) = nothing, nothing, nothing, nothing, nothing, nothing;

"""
    initialise_ecm_stats_transition(coordinates_transition_current::IntVector, coordinates_transition_lagged::IntVector, coordinates_transition_PPs::IntVector)

Initialise F, G and H.

    initialise_ecm_stats_transition(coordinates_transition_current::IntVector, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)

Initialise F, G and H (G and H to nothing)

    initialise_ecm_stats_transition(coordinates_transition_current::Nothing, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)

Initialise F, G and H to nothing.
"""
function initialise_ecm_stats_transition(coordinates_transition_current::IntVector, coordinates_transition_lagged::IntVector, coordinates_transition_PPs::IntVector)
    F = zeros(length(coordinates_transition_current), length(coordinates_transition_current));
    G = zeros(length(coordinates_transition_current), length(coordinates_transition_PPs));
    H = zeros(length(coordinates_transition_lagged), length(coordinates_transition_lagged));

    return F, G, H;
end

function initialise_ecm_stats_transition(coordinates_transition_current::IntVector, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)
    F = zeros(length(coordinates_transition_current), length(coordinates_transition_current));
    return F, nothing, nothing;
end

initialise_ecm_stats_transition(coordinates_transition_current::Nothing, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing) = nothing, nothing, nothing;

"""
    update_ecm_stats_measurement!(barrier_M::FloatMatrix, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix)

Update the ECM statistics for the measurement equation.

    update_ecm_stats_measurement!(barrier_M::FloatMatrix, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix)
    update_ecm_stats_measurement!(barrier_M::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix)
    update_ecm_stats_measurement!(barrier_M::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix)

Do not update ECM statistics for the measurement equation of models for which these calculations are not required.
"""
function update_ecm_stats_measurement!(barrier_M::FloatMatrix, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix)

    # Views
    Y_obs = @view estim.Y[ind_not_missings, t];
    Xs_view = @view Xs[coordinates_measurement_states];
    Ps_view = @view Ps[coordinates_measurement_states, coordinates_measurement_states];

    # Necessary statistics
    A_transpose = zeros(estim.n, length(ind_not_missings));
    for (i, j) in enumerate(ind_not_missings)
        A_transpose[j, i] = 1.0;
    end

    # Update ECM statistics: M
    mul!(smoother_arrays.buffer_M, A_transpose, Y_obs);
    mul!(smoother_arrays.M, smoother_arrays.buffer_M, Xs_view', 1.0, 1.0);

    # Update ECM statistics: compute N_t and O_t matrices
    mul!(smoother_arrays.buffer_N.data, A_transpose, A_transpose');
    mul!(smoother_arrays.buffer_O.data, Xs_view, Xs_view');
    smoother_arrays.buffer_O.data .+= Ps_view;

    # Store N[t] and O[t]
    smoother_arrays.N[t] = [col[:] for col in eachcol(smoother_arrays.buffer_N)];
    smoother_arrays.O[t] = [col[:] for col in eachcol(smoother_arrays.buffer_O)];
end

update_ecm_stats_measurement!(barrier_M::FloatMatrix, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix) = nothing;
update_ecm_stats_measurement!(barrier_M::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix) = nothing;
update_ecm_stats_measurement!(barrier_M::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix) = nothing;

"""
    call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix)
    call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix)
    call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix)
    call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix)

APIs to call `update_ecm_stats_measurement!` with SmootherArrays.
"""
call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix) = update_ecm_stats_measurement!(smoother_arrays.M, estim, smoother_arrays, coordinates_measurement_states, ind_not_missings, t, Xs, Ps);
call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::IntVector, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix) = update_ecm_stats_measurement!(smoother_arrays.M, estim, smoother_arrays, coordinates_measurement_states, ind_not_missings, t, Xs, Ps);
call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::IntVector, t::Int64, Xs::FloatVector, Ps::SymMatrix) = update_ecm_stats_measurement!(smoother_arrays.M, estim, smoother_arrays, coordinates_measurement_states, ind_not_missings, t, Xs, Ps);
call_update_ecm_stats_measurement!(estim::EstimSettings, smoother_arrays::SmootherArrays, coordinates_measurement_states::Nothing, ind_not_missings::Nothing, t::Int64, Xs::FloatVector, Ps::SymMatrix) = update_ecm_stats_measurement!(smoother_arrays.M, estim, smoother_arrays, coordinates_measurement_states, ind_not_missings, t, Xs, Ps);

"""
    update_ecm_stats_transition!(barrier_F::FloatMatrix, barrier_G::FloatMatrix, estim::EstimSettings, smoother_arrays::SmootherArrays, Xs_lagged::FloatVector, Ps_lagged::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::IntVector, coordinates_transition_PPs::IntVector)

Update the ECM statistics for the transition equation.

    update_ecm_stats_transition!(barrier_F::FloatMatrix, barrier_G::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, Xs_lagged::FloatVector, Ps_lagged::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)

Update F only.

    update_ecm_stats_transition!(barrier_F::Nothing, barrier_G::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, Xs_lagged::FloatVector, Ps_lagged::SymMatrix, coordinates_transition_current::Nothing, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)

Do not update ECM statistics for the transition equation of models for which these calculations are not required.
"""
function update_ecm_stats_transition!(barrier_F::FloatMatrix, barrier_G::FloatMatrix, estim::EstimSettings, smoother_arrays::SmootherArrays, Xs_lagged::FloatVector, Ps_lagged::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::IntVector, coordinates_transition_PPs::IntVector)

    # Views
    Xs_view = @view smoother_arrays.Xs_leading[coordinates_transition_current];
    Ps_view = @view smoother_arrays.Ps_leading[coordinates_transition_current, coordinates_transition_current];
    Xs_lagged_view = @view Xs_lagged[coordinates_transition_lagged];
    Ps_lagged_view = @view Ps_lagged[coordinates_transition_lagged, coordinates_transition_lagged];
    PPs_view = @view smoother_arrays.Ps_leading[coordinates_transition_current, coordinates_transition_PPs];

    # Update ECM statistics
    smoother_arrays.F .+= Xs_view*Xs_view' + Ps_view;
    smoother_arrays.G .+= Xs_view*Xs_lagged_view' + PPs_view;
    smoother_arrays.H .+= Xs_lagged_view*Xs_lagged_view' + Ps_lagged_view;
end

function update_ecm_stats_transition!(barrier_F::FloatMatrix, barrier_G::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, Xs_lagged::FloatVector, Ps_lagged::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)

    # Views
    Xs_view = @view smoother_arrays.Xs_leading[coordinates_transition_current];
    Ps_view = @view smoother_arrays.Ps_leading[coordinates_transition_current, coordinates_transition_current];

    # Update ECM statistics
    smoother_arrays.F .+= Xs_view*Xs_view' + Ps_view;
end

update_ecm_stats_transition!(barrier_F::Nothing, barrier_G::Nothing, estim::EstimSettings, smoother_arrays::SmootherArrays, Xs_lagged::FloatVector, Ps_lagged::SymMatrix, coordinates_transition_current::Nothing, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing) = nothing;

"""
    call_update_ecm_stats_transition!(estim::EstimSettings, smoother_arrays::SmootherArrays, Xs::FloatVector, Ps::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::IntVector, coordinates_transition_PPs::IntVector)
    call_update_ecm_stats_transition!(estim::EstimSettings, smoother_arrays::SmootherArrays, Xs::FloatVector, Ps::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)
    call_update_ecm_stats_transition!(estim::EstimSettings, smoother_arrays::SmootherArrays, Xs::FloatVector, Ps::SymMatrix, coordinates_transition_current::Nothing, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing)

APIs to call `update_ecm_stats_transition!` with SmootherArrays.
"""
call_update_ecm_stats_transition!(estim::EstimSettings, smoother_arrays::SmootherArrays, Xs::FloatVector, Ps::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::IntVector, coordinates_transition_PPs::IntVector) = update_ecm_stats_transition!(smoother_arrays.F, smoother_arrays.G, estim, smoother_arrays, Xs, Ps, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);
call_update_ecm_stats_transition!(estim::EstimSettings, smoother_arrays::SmootherArrays, Xs::FloatVector, Ps::SymMatrix, coordinates_transition_current::IntVector, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing) = update_ecm_stats_transition!(smoother_arrays.F, smoother_arrays.G, estim, smoother_arrays, Xs, Ps, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);
call_update_ecm_stats_transition!(estim::EstimSettings, smoother_arrays::SmootherArrays, Xs::FloatVector, Ps::SymMatrix, coordinates_transition_current::Nothing, coordinates_transition_lagged::Nothing, coordinates_transition_PPs::Nothing) = update_ecm_stats_transition!(smoother_arrays.F, smoother_arrays.G, estim, smoother_arrays, Xs, Ps, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);

"""
    reinitialise_ecm_stats!(A::Nothing)
    reinitialise_ecm_stats!(A::FloatVector)
    reinitialise_ecm_stats!(A::FloatMatrix)
    reinitialise_ecm_stats!(A::Array{VectorsArray{Float64},1})
    reinitialise_ecm_stats!(smoother_arrays::SmootherArrays)

Re-initialise ECM statistics for the following iteration.
"""
reinitialise_ecm_stats!(A::Nothing) = nothing;
reinitialise_ecm_stats!(A::FloatVector) = fill!(A, 0.0);
reinitialise_ecm_stats!(A::FloatMatrix) = fill!(A, 0.0);
reinitialise_ecm_stats!(A::SymMatrix) = fill!(A.data, 0.0);
reinitialise_ecm_stats!(A::Array{VectorsArray{Float64},1}) = nothing; # the items of A are replaced in-place in the ecm iteration, when needed - thus, empty!(A) is not needed.

function reinitialise_ecm_stats!(smoother_arrays::SmootherArrays)
    reinitialise_ecm_stats!(smoother_arrays.J1);
    reinitialise_ecm_stats!(smoother_arrays.J2);
    reinitialise_ecm_stats!(smoother_arrays.Xs_leading);
    reinitialise_ecm_stats!(smoother_arrays.Ps_leading);
    reinitialise_ecm_stats!(smoother_arrays.F);
    reinitialise_ecm_stats!(smoother_arrays.G);
    reinitialise_ecm_stats!(smoother_arrays.H);
    reinitialise_ecm_stats!(smoother_arrays.M);
    reinitialise_ecm_stats!(smoother_arrays.N);
    reinitialise_ecm_stats!(smoother_arrays.O);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Kalman smoother and CM-step for initial conditions (i.e., X0 and P0)
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    call_update_smoothing_factors!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, ind_not_missings::Union{IntVector, Nothing}, e::Union{FloatVector, Nothing}, inv_F::Union{SymMatrix, Nothing}, L::Union{FloatMatrix, Nothing})
    call_update_smoothing_factors!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, ind_not_missings::Union{IntVector, Nothing})

APIs to call `MessyTimeSeries.update_smoothing_factors!` with SmootherArrays.
"""
call_update_smoothing_factors!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, ind_not_missings::Union{IntVector, Nothing}, e::Union{FloatVector, Nothing}, inv_F::Union{SymMatrix, Nothing}, L::Union{FloatMatrix, Nothing}) = update_smoothing_factors!(sspace, status, ind_not_missings, smoother_arrays.J1, smoother_arrays.J2, e, inv_F, L);
call_update_smoothing_factors!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, ind_not_missings::Union{IntVector, Nothing}) = update_smoothing_factors!(sspace, status, ind_not_missings, smoother_arrays.J1, smoother_arrays.J2);

"""
    call_backwards_pass(Xp::FloatVector, Pp::SymMatrix, smoother_arrays::SmootherArrays)
    call_backwards_pass(Pp::SymMatrix, smoother_arrays::SmootherArrays)

APIs to call `MessyTimeSeries.backwards_pass` with SmootherArrays.
"""
call_backwards_pass(Xp::FloatVector, Pp::SymMatrix, smoother_arrays::SmootherArrays) = backwards_pass(Xp, Pp, smoother_arrays.J1);
call_backwards_pass(Pp::SymMatrix, smoother_arrays::SmootherArrays) = backwards_pass(Pp, smoother_arrays.J2);

"""
    ksmoother_ecm_iteration!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, t::Int64)

Kalman smoother iteration of `ksmoother_ecm!`
"""
function ksmoother_ecm_iteration!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, t::Int64)

    # Pointers
    Xp = status.history_X_prior[t];
    Pp = status.history_P_prior[t];
    e = status.history_e[t];
    inv_F = status.history_inv_F[t];
    L = status.history_L[t];

    # Handle missing observations
    ind_not_missings = find_observed_data(sspace, t);

    # Smoothed estimates for t
    call_update_smoothing_factors!(sspace, status, smoother_arrays, ind_not_missings, e, inv_F, L);
    Xs = call_backwards_pass(Xp, Pp, smoother_arrays)::FloatVector;
    Ps = call_backwards_pass(Pp, smoother_arrays)::SymMatrix;

    # Return smoothed estimates for t
    return Xs, Ps, ind_not_missings;
end

"""
    cm_step_X0_P0!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, coordinates_transition_P0::IntVector)

CM-step for X0 and P0.
"""
function cm_step_X0_P0!(sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, coordinates_transition_P0::IntVector)

    # CM-step for X0
    mul!(sspace.X0, sspace.P0, smoother_arrays.J1, 1.0, 1.0);

    # CM-step for sspace.P0
    mul!(status.online_status.buffer_J2, sspace.P0, Array(smoother_arrays.J2));         # Array(...) helps speeding up mul!(...) while keeping J2 symmetric (i.e., good trade-off wrt other options incl. ".data")
    mul!(status.online_status.buffer_m_m, status.online_status.buffer_J2, sspace.P0);
    for ij in coordinates_transition_P0
        sspace.P0.data[ij] -= status.online_status.buffer_m_m[ij];
    end
end

"""
    ksmoother_ecm!(estim::EstimSettings, sspace::KalmanSettings, status::SizedKalmanStatus, coordinates_measurement_states::Union{IntVector, Nothing}, coordinates_transition_current::Union{IntVector, Nothing}, coordinates_transition_lagged::Union{IntVector, Nothing}, coordinates_transition_PPs::Union{IntVector, Nothing}, coordinates_transition_P0::IntVector)

Kalman smoother: RTS smoother from the last evaluated time period in status to t==0.

The smoother is implemented following the approach proposed in Durbin and Koopman (2012).

This instance of the smoother returns the ECM statistics and updates the initial conditions in KalmanSettings.

# Arguments
- `estim`: EstimSettings struct
- `sspace`: KalmanSettings struct
- `status`: SizedKalmanStatus struct
- `smoother_arrays`: SmootherArrays struct
- `coordinates_measurement_states`: indices describing the states needed to construct M, N, O (or nothing)
- `coordinates_transition_current`, `coordinates_transition_lagged` and `coordinates_transition_PPs`: indices describing the elements of interest needed to construct F, G and H (or nothing)
- `coordinates_transition_P0`: index describing which element of P0 should be re-estimated in the cm-step
"""
function ksmoother_ecm!(estim::EstimSettings, sspace::KalmanSettings, status::SizedKalmanStatus, smoother_arrays::SmootherArrays, coordinates_measurement_states::Union{IntVector, Nothing}, coordinates_transition_current::Union{IntVector, Nothing}, coordinates_transition_lagged::Union{IntVector, Nothing}, coordinates_transition_PPs::Union{IntVector, Nothing}, coordinates_transition_P0::IntVector)

    # Loop over t (from status.t-1 to 1)
    for t=status.online_status.t:-1:1

        # Kalman smoother iteration
        Xs, Ps, ind_not_missings = ksmoother_ecm_iteration!(sspace, status, smoother_arrays, t);

        # Update ECM statistics for the measurement equation
        call_update_ecm_stats_measurement!(estim, smoother_arrays, coordinates_measurement_states, ind_not_missings, t, Xs, Ps);

        #=
        Update ECM statistics for the transition equation - please note that:
            1. the following if-statement makes sure to have a valid value for the *_leading estimators;
            2. implementing the update this way is beneficial from a computational standpoint, but `update_ecm_stats_transition!` needs to be called an additional time outside the loop to finalise the calculations.
        =#

        if t < status.online_status.t
            call_update_ecm_stats_transition!(estim, smoother_arrays, Xs, Ps, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);
        end

        # Update Xs_leading and Ps_leading
        copyto!(smoother_arrays.Xs_leading, Xs);
        copyto!(smoother_arrays.Ps_leading.data, Ps.data);
    end

    # Compute smoothed estimates for t==0
    call_update_smoothing_factors!(sspace, status, smoother_arrays, nothing);

    # CM-step for X0 and P0
    cm_step_X0_P0!(sspace, status, smoother_arrays, coordinates_transition_P0);

    # Finalise calculations of the ECM statistics for the transition equation
    call_update_ecm_stats_transition!(estim, smoother_arrays, sspace.X0, sspace.P0, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
CM-step
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    turbo_dot(x::AbstractVector{Float64}, y::AbstractVector{Float64})

Dot product.

    turbo_dot(x::AbstractVector{Float64}, A::AbstractMatrix{Float64}, y::AbstractVector{Float64})

Generalised dot product.
"""
@inline function turbo_dot(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    s = 0.0;
    @turbo for i in eachindex(x)
        s += x[i]*y[i];
    end
    return s;
end

@inline function turbo_dot(x::AbstractVector{Float64}, A::AbstractMatrix{Float64}, y::AbstractVector{Float64})
    s = 0.0;
    for j in eachindex(y)
        @inbounds yj = y[j];
        temp = 0.0;
        if !iszero(yj)
            @turbo for i in eachindex(x)
                temp += x[i]*A[i,j];
            end
            s += temp*yj;
        end
    end
    return s;
end

"""
    cm_step_time_loop(sspace::KalmanSettings, B_star::SubArray{Float64}, ij::CartesianIndex{2}, N::Array{VectorsArray{Float64},1}, O::Array{VectorsArray{Float64},1})

Compute numerator and denominator to compute the (i,j)-th CM update for the measurement equation coefficients in `cm_step!`.
"""
function cm_step_time_loop(sspace::KalmanSettings, B_star::SubArray{Float64}, ij::CartesianIndex{2}, N::Array{VectorsArray{Float64},1}, O::Array{VectorsArray{Float64},1})

    # Coordinates
    i,j = ij.I;

    # Store transpose of B_star in Array format and B_{ij} to speed-up calculations
    B_star_transpose = B_star' |> FloatMatrix;
    B_star_ij = B_star[ij];

    # Memory pre-allocation for ij sums
    sum_numerator = 0.0;
    sum_denominator = 0.0;

    for t=1:sspace.Y.T

        # Proceed if N_t and O_t are not #undef (the points in time when at least one series is observed)
        if isassigned(N, t) && isassigned(O, t)

            # Pointers
            N_t = N[t];
            O_t = O[t];
            N_it = N_t[i];
            O_jt = O_t[j];

            # Shortcut
            @inbounds NO_ij_t = N_it[i]*O_jt[j];

            # Update numerator
            sum_numerator += turbo_dot(O_jt, B_star_transpose, N_it); # this order gives a faster performance, since N_it includes a significant number of zeros!
            sum_numerator -= NO_ij_t*B_star_ij;

            # Update `sum_denominator`
            sum_denominator += NO_ij_t;
        end
    end

    return sum_numerator, sum_denominator, i, j;
end

"""
    call_cm_step!(estim::EstimSettings, sspace::KalmanSettings, smoother_arrays::SmootherArrays, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector)
    call_cm_step!(estim::EstimSettings, sspace::KalmanSettings, smoother_arrays::SmootherArrays, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing)
    call_cm_step!(estim::EstimSettings, sspace::KalmanSettings, smoother_arrays::SmootherArrays, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector)

APIs to call `cm_step!` with SmootherArrays.
"""
call_cm_step!(estim::EstimSettings, sspace::KalmanSettings, smoother_arrays::SmootherArrays, B_star::SubArray{Float64}, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::CoordinatesVector) = cm_step!(estim, sspace, B_star, C_star, Q_view, smoother_arrays.F, smoother_arrays.G, smoother_arrays.H, smoother_arrays.M, smoother_arrays.N, smoother_arrays.O, coordinates_free_params_B, coordinates_free_params_C);
call_cm_step!(estim::EstimSettings, sspace::KalmanSettings, smoother_arrays::SmootherArrays, B_star::SubArray{Float64}, C_star::Nothing, Q_view::SubArray{Float64}, coordinates_free_params_B::CoordinatesVector, coordinates_free_params_C::Nothing) = cm_step!(estim, sspace, B_star, C_star, Q_view, smoother_arrays.F, smoother_arrays.G, smoother_arrays.H, smoother_arrays.M, smoother_arrays.N, smoother_arrays.O, coordinates_free_params_B, coordinates_free_params_C);
call_cm_step!(estim::EstimSettings, sspace::KalmanSettings, smoother_arrays::SmootherArrays, B_star::Nothing, C_star::SubArray{Float64}, Q_view::SubArray{Float64}, coordinates_free_params_B::Nothing, coordinates_free_params_C::CoordinatesVector) = cm_step!(estim, sspace, B_star, C_star, Q_view, smoother_arrays.F, smoother_arrays.G, smoother_arrays.H, smoother_arrays.M, smoother_arrays.N, smoother_arrays.O, coordinates_free_params_B, coordinates_free_params_C);

"""
    enforce_causality_and_invertibility_write!(companion_new::FloatMatrix, companion_proposal::FloatMatrix, params_new::Nothing)
    enforce_causality_and_invertibility_write!(companion_new::AbstractArray{Float64,2}, companion_proposal::FloatMatrix, params_new::FloatMatrix)

Barrier function for `enforce_causality_and_invertibility!` based on @turbo. Replace inplace either `companion_new` or `params_new` depending on whether the latter is `nothing` or not.

    enforce_causality_and_invertibility_write!(companion_new::AbstractArray{Float64,2}, companion_proposal::AbstractArray{Float64,2}, params_new::Nothing)
    enforce_causality_and_invertibility_write!(companion_new::AbstractArray{Float64,2}, companion_proposal::AbstractArray{Float64,2}, params_new::AbstractArray{Float64,2})

Barrier function for `enforce_causality_and_invertibility!`. Replace inplace either `companion_new` or `params_new` depending on whether the latter is `nothing` or not.
"""
@inline function enforce_causality_and_invertibility_write!(companion_new::FloatMatrix, companion_proposal::FloatMatrix, params_new::Nothing)
    @turbo for ij in eachindex(companion_new)
        companion_new[ij] = companion_proposal[ij];
    end
end

@inline function enforce_causality_and_invertibility_write!(companion_new::AbstractArray{Float64,2}, companion_proposal::FloatMatrix, params_new::FloatMatrix)
    @turbo for j in axes(params_new, 2), i in axes(params_new, 1)
        params_new[i,j] = companion_proposal[i,j];
    end
end

function enforce_causality_and_invertibility_write!(companion_new::AbstractArray{Float64,2}, companion_proposal::AbstractArray{Float64,2}, params_new::Nothing)
    for ij in eachindex(companion_new)
        @inbounds companion_new[ij] = companion_proposal[ij];
    end
end

function enforce_causality_and_invertibility_write!(companion_new::AbstractArray{Float64,2}, companion_proposal::AbstractArray{Float64,2}, params_new::AbstractArray{Float64,2})
    for j in axes(params_new, 2), i in axes(params_new, 1)
        @inbounds params_new[i,j] = companion_proposal[i,j];
    end
end

"""
    enforce_causality_and_invertibility!(companion_new::AbstractArray{Float64,2}, companion_old::AbstractArray{Float64,2}, params_new::Union{Nothing, AbstractArray{Float64,2}}=nothing)

Shrink `companion_new` towards `companion_old` (when needed) to ensure it is causal / invertible (via grid search). If `params_new` is not `nothing` the function output replaces it inplace instead. 

    enforce_causality_and_invertibility!(params_new::AbstractArray{Float64,2})

Compute the companion form of `params_new` and shrink it towards zero (when needed) to ensure it is causal / invertible (via grid search).
"""
function enforce_causality_and_invertibility!(companion_new::AbstractArray{Float64,2}, companion_old::AbstractArray{Float64,2}, params_new::Union{Nothing, AbstractArray{Float64,2}}=nothing)

    # Loop over grid of weights
    for weight=1.0:-0.1:0.0

        # Compute weighted cycle
        companion_proposal = (weight*companion_new + (1-weight)*companion_old)::FloatMatrix;

        # Compute corresponding largest eigenvalue
        ith_eigmax = maximum(abs.(eigvals(companion_proposal)));

        # Keep current setting and adjust
        if (ith_eigmax <= 0.98) && (weight != 1)
            enforce_causality_and_invertibility_write!(companion_new, companion_proposal, params_new);
            break;

        # No adjustment needed
        elseif (ith_eigmax <= 0.98) && (weight == 1)
            break;
        end
    end
end

function enforce_causality_and_invertibility!(params_new::AbstractArray{Float64,2})
    companion_old = companion_form(zeros(size(params_new)), extended=false);
    companion_new = companion_form(params_new, extended=false);
    enforce_causality_and_invertibility!(companion_new, companion_old, params_new);
end

#=
--------------------------------------------------------------------------------------------------------------------------------
Optimisation
--------------------------------------------------------------------------------------------------------------------------------
=#

"""
    sspace_representation(estim::EstimSettings, B_star::FloatMatrix, C_star::FloatMatrix, Q::SymMatrix; extended=true)

Return the default state-space coefficients. If extended=true, `n` additional states at the end for the PPs calculation proposed in Watson and Engle (1983).
"""
function sspace_representation(estim::EstimSettings, B_star::FloatMatrix, C_star::FloatMatrix, Q::SymMatrix; extended=true)
    
    # Observation equation (fixed) coefficients
    B = [Matrix(1.0I, estim.n, estim.n) B_star zeros(estim.n*extended, estim.n*extended)];
    R = Symmetric(Matrix(estim.ε * I, estim.n, estim.n));
    
    # Transition equation
    C = companion_form(C_star, extended=extended);
    D = zeros(estim.m+estim.n*extended, estim.n);
    D[1:estim.n, :] = Matrix(1.0I, estim.n, estim.n);

    # Return output
    return B, R, C, D, Q;
end

"""
    update_sspace_data!(sspace::KalmanSettings, output_sspace_data::Union{FloatMatrix, JMatrix{Float64}})

Update `sspace.Y` with `output_sspace_data`.

    update_sspace_data!(sspace::KalmanSettings, output_sspace_data::Nothing)

Return `nothing`.
"""
function update_sspace_data!(sspace::KalmanSettings, output_sspace_data::Union{FloatMatrix, JMatrix{Float64}})
    sspace.Y.data = output_sspace_data;
    sspace.Y.T = size(output_sspace_data, 2);
end

update_sspace_data!(sspace::KalmanSettings, output_sspace_data::Nothing) = nothing;

"""
    ecm(estim::EstimSettings)

Run the ECM algorithm in Pellegrino (2022).

# Arguments
- `estim`: settings used for the estimation
- `output_sspace_data`: Optional argument. If specified, it is used as the output state space data. Otherwise, estim.Y is used instead.

# References
Pellegrino (2022)
"""
function ecm(estim::EstimSettings; output_sspace_data::Union{FloatMatrix, JMatrix{Float64}, Nothing}=nothing)

    # Check inputs
    check_model_bounds(estim);

    #=
    The state vector includes additional terms with respect to standard companion form representations.
    This is to estimate the lag-one covariance smoother as in Watson and Engle (1983).
    =#

    # Initialise state-space
    verb_message(estim.verb, "ecm > initialisation");
    sspace, B_star, C_star, Q_view, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs, coordinates_transition_P0, coordinates_free_params_B, coordinates_free_params_C = initialise(estim);

    # Pre-allocate memory for Kalman filter and smoother
    status = SizedKalmanStatus(sspace.Y.T);
    smoother_arrays = SmootherArrays(estim, sspace, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs);

    # ECM controls
    ecm_params_old = vec_ecm_params(estim, B_star, C_star, Q_view, coordinates_free_params_B, coordinates_free_params_C);
    ecm_params_new = similar(ecm_params_old);
    params_rel_change = similar(ecm_params_old);

    # Run ECM
    for iter=1:estim.max_iter

        # Run Kalman filter
        kfilter_full_sample!(sspace, status);

        # Compute Kalman stats
        ksmoother_ecm!(estim, sspace, status, smoother_arrays, coordinates_measurement_states, coordinates_transition_current, coordinates_transition_lagged, coordinates_transition_PPs, coordinates_transition_P0);

        # CM-step
        call_cm_step!(estim, sspace, smoother_arrays, B_star, C_star, Q_view, coordinates_free_params_B, coordinates_free_params_C);

        if iter > estim.prerun
    
            # Update ecm_params_new
            vec_ecm_params!(estim, ecm_params_new, B_star, C_star, Q_view, coordinates_free_params_B, coordinates_free_params_C);

            # Relative change criterion
            update_relative_change!(estim, params_rel_change, ecm_params_new, ecm_params_old);

            # Stop when the ECM algorithm converges
            if iter > estim.prerun+1
                if ecm_converged(estim, params_rel_change, iter)
                    verb_message(estim.verb, "ecm > converged!\n");
                    break;
                end
            end

            # Store current run information
            copyto!(ecm_params_old, ecm_params_new);

        else
            verb_message(estim.verb, "ecm > prerun $iter (out of $(estim.prerun))");
        end

        reinitialise_ecm_stats!(smoother_arrays);
    end

    # Final operations
    update_sspace_data!(sspace, output_sspace_data); # TBD: Use compress representation
    sspace.Q.data .= Array(Q_view); # update Q (for internal consistency)

    # Return output
    return sspace;
end