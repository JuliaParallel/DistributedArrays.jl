using Base.Test

using DistributedArrays

# add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

@everywhere importall DistributedArrays
@everywhere importall DistributedArrays.SPMD

@everywhere srand(1234 + myid())

const MYID = myid()
const OTHERIDS = filter(id-> id != MYID, procs())[rand(1:(nprocs()-1))]

function check_leaks()
    if length(DistributedArrays.refs) > 0
        sleep(0.1)  # allow time for any cleanup to complete and test again
        length(DistributedArrays.refs) > 0 && warn("Probable leak of ", length(DistributedArrays.refs), " darrays")
    end
end

include("darray.jl")
include("spmd.jl")

