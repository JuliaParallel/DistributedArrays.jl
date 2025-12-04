using Test
using Distributed
using DistributedArrays

# Disable scalar indexing to avoid falling back on generic methods
# for AbstractArray
DistributedArrays.allowscalar(false)

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_THREADS))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

@everywhere using Distributed
@everywhere using DistributedArrays
@everywhere using DistributedArrays.SPMD
@everywhere using Random
@everywhere using LinearAlgebra

@everywhere Random.seed!(1234 + myid())

const MYID = myid()
const OTHERIDS = filter(id-> id != MYID, procs())[rand(1:(nprocs()-1))]

function check_leaks()
    nrefs = @lock DistributedArrays.REFS.lock length(DistributedArrays.REFS.data)
    if !iszero(nrefs)
        sleep(0.1)  # allow time for any cleanup to complete and test again
        nrefs = @lock DistributedArrays.REFS.lock length(DistributedArrays.REFS.data)
        if !iszero(nrefs)
            @warn("Probable leak of ", nrefs, " darrays")
        end
    end
end

include("aqua.jl")
include("explicit_imports.jl")
include("darray.jl")
include("spmd.jl")

