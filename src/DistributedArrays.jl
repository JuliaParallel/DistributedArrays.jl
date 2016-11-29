__precompile__(true)

module DistributedArrays

using Primes
using Primes: factor

importall Base
import Base.Callable
import Base.BLAS: axpy!

# DArray exports
export (.+), (.-), (.*), (./), (.%), (.<<), (.>>), div, mod, rem, (&), (|), ($)
export DArray, SubDArray, SubOrDArray, @DArray
export dzeros, dones, dfill, drand, drandn, distribute, localpart, localindexes, ppeval, samedist

# non-array distributed data
export ddata, gather

# immediate release of localparts
export close, d_closeall

include("darray.jl")
include("core.jl")
include("serialize.jl")
include("mapreduce.jl")
include("linalg.jl")
include("sort.jl")

include("spmd.jl")

end # module
