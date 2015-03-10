using Base.Test

n = min(8, CPU_CORES)
n > 1 && addprocs(n; exeflags=`--check-bounds=yes`)

@everywhere using DistributedArrays
@everywhere blas_set_num_threads(1)

include("darray.jl")
