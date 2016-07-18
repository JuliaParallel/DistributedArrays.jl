# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

using Compat
import Compat.view
# It should only be necessary to have FactCheck loaded on the master process, but
# https://github.com/JuliaLang/julia/issues/15766. Move back to top when bug is fixed.
using FactCheck
using DistributedArrays
using StatsBase # for fit(Histogram, ...)
@everywhere using StatsBase # because exported functions are not exported on workers with using

@everywhere srand(123 + myid())

include("darray.jl")

FactCheck.exitstatus()
