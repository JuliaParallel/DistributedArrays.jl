using Base.Test

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end

@assert nprocs() > 3
@assert nworkers() >= 3

@everywhere using DistributedArrays
@everywhere blas_set_num_threads(1)

include("darray.jl")
