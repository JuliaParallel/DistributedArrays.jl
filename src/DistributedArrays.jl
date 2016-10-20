__precompile__(true)

module DistributedArrays

using Primes
using Primes: factor

importall Base
import Base.Callable
import Base.BLAS: axpy!

export (.+), (.-), (.*), (./), (.%), (.<<), (.>>), div, mod, rem, (&), (|), ($)
export Distributed, SubDistributed, SubOrDistributed, @Distributed
export dzeros, dones, dfill, drand, drandn, distribute, localpart, localindexes, ppeval, samedist
export close, darray_closeall

include("core.jl")
include("serialize.jl")
include("mapreduce.jl")
include("linalg.jl")
include("sort.jl")

end # module
