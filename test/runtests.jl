using Base.Test

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

using DistributedArrays
using StatsBase # for fit(Histogram, ...)
@everywhere using StatsBase # because exported functions are not exported on workers with using

@everywhere srand(1234 + myid())

include("darray.jl")
