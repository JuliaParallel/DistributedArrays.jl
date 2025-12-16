module SparseArraysExt

using DistributedArrays: DArray, SubDArray, SubOrDArray, localpart
using DistributedArrays.Distributed: remotecall_fetch
using SparseArrays: SparseArrays, nnz

function SparseArrays.nnz(A::DArray)
    B = asyncmap(A.pids) do p
        remotecall_fetch(nnz∘localpart, p, A)
    end
    return reduce(+, B)
end

# Fix method ambiguities
# TODO: Improve efficiency?
Base.copyto!(dest::SubOrDArray{<:Any,2}, src::SparseArrays.AbstractSparseMatrixCSC) = copyto!(dest, Matrix(src))
@static if isdefined(SparseArrays, :CHOLMOD)
    Base.copyto!(dest::SubOrDArray, src::SparseArrays.CHOLMOD.Dense) = copyto!(dest, Array(src))
    Base.copyto!(dest::SubOrDArray{T}, src::SparseArrays.CHOLMOD.Dense{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}} = copyto!(dest, Array(src))
    Base.copyto!(dest::SubOrDArray{T,2}, src::SparseArrays.CHOLMOD.Dense{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}} = copyto!(dest, Array(src))
end

# Fix method ambiguities
for T in (:DArray, :SubDArray)
    @eval begin
        Base.:(==)(d1::$T{<:Any,1}, d2::SparseArrays.ReadOnly) = d1 == parent(d2)
        Base.:(==)(d1::SparseArrays.ReadOnly, d2::$T{<:Any,1}) = parent(d1) == d2
    end
end

end
