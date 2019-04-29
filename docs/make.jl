using Documenter, DistributedArrays

makedocs(
    modules = [DistributedArrays],
    format = Documenter.HTML(),
    sitename = "DistributedArrays.jl",
    pages = [
        "Introduction" => "index.md"
        "API" => "api.md"
    ],
    doctest = true
)

deploydocs(
    repo = "github.com/JuliaParallel/DistributedArrays.jl.git",
)
