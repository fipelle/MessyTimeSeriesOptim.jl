@testset "Tests on enforce_causality_and_invertibility!" begin

    # Setup
    A = [0.5 0.4 0.2 0.1; 0.3 -0.2 0.1 0.1];
    B = 1.25 * [0.5 0.4 0.2 0.1; 0.3 -0.2 0.1 0.1];
    A_companion = companion_form(A, extended=false);
    B_companion = companion_form(B, extended=false);
    A_view = @view A_companion[1:2, :];
    B_view = @view B_companion[1:2, :];
    benchmark = [0.5 0.4 0.2 0.1; 0.3 -0.2 0.1 0.1];

    # Test 1a: vector input does not change
    MessyTimeSeriesOptim.enforce_causality_and_invertibility!(A);
    @test A == benchmark;

    # Test 1b: vector input changes
    MessyTimeSeriesOptim.enforce_causality_and_invertibility!(B);
    @test round.(B, digits=16) == benchmark;

    # Test 2a: view input does not change
    MessyTimeSeriesOptim.enforce_causality_and_invertibility!(A_view);
    @test A_view == benchmark;

    # Test 2b: view input changes
    MessyTimeSeriesOptim.enforce_causality_and_invertibility!(B_view);
    @test round.(B_view, digits=16) == benchmark;

    # View consequences on the original companion forms
    @test round.(B_companion, digits=16) == A_companion;

    # Reset the companion forms to perform final tst
    A_companion = companion_form([0.5 0.4 0.2 0.1; 0.3 -0.2 0.1 0.1], extended=false);
    B_companion = companion_form(1.25 * [0.5 0.4 0.2 0.1; 0.3 -0.2 0.1 0.1], extended=false);
    new_benchmark = 0.7*A_companion + 0.3*B_companion;

    # Test on companion forms
    MessyTimeSeriesOptim.enforce_causality_and_invertibility!(B_companion, A_companion);
    @test B_companion == new_benchmark;
end