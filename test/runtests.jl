if VERSION >= v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

using Compat
import Compat.view
using DistributedArrays
using StatsBase # for fit(Histogram, ...)
@everywhere using StatsBase # because exported functions are not exported on workers with using

@everywhere srand(1234 + myid())

@testset "DArray tests" begin
    include("darray.jl")
end
