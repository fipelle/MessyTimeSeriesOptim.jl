using Documenter, MessyTimeSeriesOptim;
makedocs(
    sitename="MessyTimeSeriesOptim.jl",     
    pages = [
        "index.md",
        "Estimation"  => ["man/estimation.md"],
        "Validation"  => ["man/validation.md"],
    ]
);

#=
deploydocs(
    repo = "github.com/fipelle/MessyTimeSeries.jl.git",
)
=#