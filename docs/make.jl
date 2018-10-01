using Documenter, DistributedArrays

makedocs(
    modules = [DistributedArrays],
    format = :html,
    sitename = "DistributedArrays.jl",
    pages = [
        "Introduction" => "index.md"
        "API" => "api.md"
    ],
    doctest = true
)

deploydocs(
    repo = "github.com/JuliaParallel/DistributedArrays.jl.git",
    julia = "1.0",
    # no need to build anything here, re-use output of `makedocs`
    target = "build",
    deps = nothing,
    make = nothing
)
