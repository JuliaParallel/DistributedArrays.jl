using FactCheck

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

using DistributedArrays
@everywhere blas_set_num_threads(1)
@everywhere srand(123 + myid())

include("darray.jl")

FactCheck.exitstatus()
