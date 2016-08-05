using Base.Test

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

using DistributedArrays

@everywhere srand(1234 + myid())

include("darray.jl")
