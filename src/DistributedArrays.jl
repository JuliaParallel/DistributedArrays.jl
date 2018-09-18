__precompile__()

module DistributedArrays

using Distributed
using Serialization
using LinearAlgebra
using Statistics

import Base: +, -, *, div, mod, rem, &, |, xor
import Base.Callable
import LinearAlgebra: axpy!, dot, norm

import Primes
import Primes: factor

# DArray exports
export DArray, SubDArray, SubOrDArray, @DArray
export dzeros, dones, dfill, drand, drandn, distribute, localpart, localindices, ppeval

# non-array distributed data
export ddata, gather

# immediate release of localparts
export close, d_closeall

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
