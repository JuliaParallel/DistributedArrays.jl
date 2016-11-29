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

# On 0.6, @testset does not display the test description automatically anymore.
function print_test_desc(t, n=0)
    VERSION < v"0.6-" && return

    println(repeat(" ", n), "Passed : ", t.description)
    for t2 in t.results
        if isa(t2, Base.Test.DefaultTestSet)
            print_test_desc(t2, n+2)
        end
    end
end

function check_leaks(t=nothing)
    if length(DistributedArrays.refs) > 0
        sleep(0.1)  # allow time for any cleanup to complete and test again
        length(DistributedArrays.refs) > 0 && warn("Probable leak of ", length(DistributedArrays.refs), " darrays")
    end

    isa(t, Base.Test.DefaultTestSet) && print_test_desc(t)
end

include("darray.jl")
include("spmd.jl")

