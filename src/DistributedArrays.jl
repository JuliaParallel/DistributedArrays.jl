module DistributedArrays

using Base: Callable
using Base.Broadcast: BroadcastStyle, Broadcasted

using Distributed: Distributed, RemoteChannel, Future, myid, nworkers, procs, remotecall, remotecall_fetch, remotecall_wait, worker_id_from_socket, workers
using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, I, Transpose, adjoint, adjoint!, axpy!, dot, lmul!, mul!, norm, rmul!, transpose, transpose!
using Random: Random, rand!
using Serialization: Serialization, AbstractSerializer, deserialize, serialize

using Primes: factor

import SparseArrays

# DArray exports
export DArray, SubDArray, SubOrDArray, @DArray
export dzeros, dones, dfill, drand, drandn, distribute, localpart, localindices, ppeval

# non-array distributed data
export ddata, gather

# immediate release of localparts
export d_closeall

include("darray.jl")
include("core.jl")
include("serialize.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("linalg.jl")
include("sort.jl")

include("spmd.jl")
export SPMD

end # module
