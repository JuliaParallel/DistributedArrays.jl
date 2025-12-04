module SparseArraysExt

using DistributedArrays: DArray, localpart
using DistributedArrays.Distributed: remotecall_fetch
using SparseArrays: SparseArrays, nnz

function SparseArrays.nnz(A::DArray)
    B = asyncmap(A.pids) do p
        remotecall_fetch(nnz∘localpart, p, A)
    end
    return reduce(+, B)
end

end
