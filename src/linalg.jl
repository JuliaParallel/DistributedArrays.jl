function Base.copy(Dadj::Adjoint{T,<:DArray{T,2}}) where T
    D = parent(Dadj)
    DArray(reverse(size(D)), procs(D)) do I
        lp = Array{T}(undef, map(length, I))
        rp = convert(Array, D[reverse(I)...])
        adjoint!(lp, rp)
    end
end

function Base.copy(Dtr::Transpose{T,<:DArray{T,2}}) where T
    D = parent(Dtr)
    DArray(reverse(size(D)), procs(D)) do I
        lp = Array{T}(undef, map(length, I))
        rp = convert(Array, D[reverse(I)...])
        transpose!(lp, rp)
    end
end

const DVector{T,A} = DArray{T,1,A}
const DMatrix{T,A} = DArray{T,2,A}

# Level 1

function axpy!(α, x::DArray, y::DArray)
    if length(x) != length(y)
        throw(DimensionMismatch("vectors must have same length"))
    end
    @sync for p in procs(y)
        @async remotecall_wait(p) do
            axpy!(α, localpart(x), localpart(y))
        end
    end
    return y
end

function dot(x::DVector, y::DVector)
    if length(x) != length(y)
        throw(DimensionMismatch(""))
    end

    results = asyncmap(procs(x)) do p
        remotecall_fetch((x, y) -> dot(localpart(x), makelocal(y, localindices(x)...)), p, x, y)
    end
    return reduce(+, results)
end

function norm(x::DArray, p::Real = 2)
    results = asyncmap(procs(x)) do pp
        remotecall_fetch(() -> norm(localpart(x), p), pp)
    end
    return norm(results, p)
end

function LinearAlgebra.rmul!(A::DArray, x::Number)
    @sync for p in procs(A)
        @async remotecall_wait((A,x)->rmul!(localpart(A), x), p, A, x)
    end
    return A
end

# Level 2
function add!(dest, src, scale = one(dest[1]))
    if length(dest) != length(src)
        throw(DimensionMismatch("source and destination arrays must have same number of elements"))
    end
    if scale == one(scale)
        @simd for i = eachindex(dest)
            @inbounds dest[i] += src[i]
        end
    else
        @simd for i = eachindex(dest)
            @inbounds dest[i] += scale*src[i]
        end
    end
    return dest
end

function LinearAlgebra.mul!(y::DVector, A::DMatrix, x::AbstractVector, α::Number = 1, β::Number = 0)

    # error checks
    if size(A, 2) != length(x)
        throw(DimensionMismatch(""))
    end
    if y.cuts[1] != A.cuts[1]
        throw(ArgumentError("cuts of output vector must match cuts of first dimension of matrix"))
    end

    # Multiply on each tile of A
    R = Array{Future}(undef, size(A.pids))
    for j = 1:size(A.pids, 2)
        xj = x[A.cuts[2][j]:A.cuts[2][j + 1] - 1]
        for i = 1:size(A.pids, 1)
            R[i,j] = remotecall(procs(A)[i,j]) do
                localpart(A)*convert(localtype(x), xj)
            end
        end
    end

    # Scale y if necessary
    if β != one(β)
        asyncmap(procs(y)) do p
            remotecall_wait(p) do
                if !iszero(β)
                    rmul!(localpart(y), β)
                else
                    fill!(localpart(y), 0)
                end
            end
        end
    end

    # Update y
    @sync for i = 1:size(R, 1)
        p = y.pids[i]
        for j = 1:size(R, 2)
            rij = R[i,j]
            @async remotecall_wait(() -> add!(localpart(y), fetch(rij), α), p)
        end
    end

    return y
end

function LinearAlgebra.mul!(y::DVector, adjA::Adjoint{<:Number,<:DMatrix}, x::AbstractVector, α::Number = 1, β::Number = 0)

    A = parent(adjA)

    # error checks
    if size(A, 1) != length(x)
        throw(DimensionMismatch(""))
    end
    if y.cuts[1] != A.cuts[2]
        throw(ArgumentError("cuts of output vector must match cuts of second dimension of matrix"))
    end

    # Multiply on each tile of A
    R = Array{Future}(undef, reverse(size(A.pids)))
    for j = 1:size(A.pids, 1)
        xj = x[A.cuts[1][j]:A.cuts[1][j + 1] - 1]
        for i = 1:size(A.pids, 2)
            R[i,j] = remotecall(() -> localpart(A)'*convert(localtype(x), xj), procs(A)[j,i])
        end
    end

    # Scale y if necessary
    if β != one(β)
        @sync for p in procs(y)
            @async remotecall_wait(p) do
                if !iszero(β)
                    rmul!(localpart(y), β)
                else
                    fill!(localpart(y), 0)
                end
            end
        end
    end

    # Update y
    @sync for i = 1:size(R, 1)
        p = y.pids[i]
        for j = 1:size(R, 2)
            rij = R[i,j]
            @async remotecall_wait(() -> add!(localpart(y), fetch(rij), α), p)
        end
    end
    return y
end

function LinearAlgebra.lmul!(D::Diagonal, DA::DMatrix)
    d = D.diag
    s = verified_destination_serializer(procs(DA), size(DA.indices)) do pididx
        d[DA.indices[pididx][1]]
    end
    map_localparts!(DA) do lDA
        lmul!(Diagonal(localpart(s)), lDA)
    end
end

function LinearAlgebra.rmul!(DA::DMatrix, D::Diagonal)
    d = D.diag
    s = verified_destination_serializer(procs(DA), size(DA.indices)) do pididx
        d[DA.indices[pididx][2]]
    end
    map_localparts!(DA) do lDA
        rmul!(lDA, Diagonal(localpart(s)))
    end
end

# Level 3
function _matmatmul!(C::DMatrix, A::DMatrix, B::AbstractMatrix, α::Number, β::Number, tA)
    # error checks
    Ad1, Ad2 = (tA == 'N') ? (1,2) : (2,1)
    mA, nA = (size(A, Ad1), size(A, Ad2))
    mB, nB = size(B)
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA, $nA), matrix B has dimensions ($mB, $nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs ($mA, $nB)"))
    end
    if C.cuts[1] != A.cuts[Ad1]
        throw(ArgumentError("cuts of the first dimension of the output matrix must match cuts of dimension $Ad1 of the first input matrix"))
    end

    # Multiply on each tile of A
    if tA == 'N'
        R = Array{Future}(undef, size(procs(A))..., size(procs(C), 2))
    else
        R = Array{Future}(undef, reverse(size(procs(A)))..., size(procs(C), 2))
    end
    for j = 1:size(A.pids, Ad2)
        for k = 1:size(C.pids, 2)
            Acuts = A.cuts[Ad2]
            Ccuts = C.cuts[2]
            Bjk = B[Acuts[j]:Acuts[j + 1] - 1, Ccuts[k]:Ccuts[k + 1] - 1]
            for i = 1:size(A.pids, Ad1)
                p = (tA == 'N') ? procs(A)[i,j] : procs(A)[j,i]
                R[i,j,k] = remotecall(p) do
                    if tA == 'T'
                        return transpose(localpart(A))*convert(localtype(B), Bjk)
                    elseif tA == 'C'
                        return adjoint(localpart(A))*convert(localtype(B), Bjk)
                    else
                        return localpart(A)*convert(localtype(B), Bjk)
                    end
                end
            end
        end
    end

    # Scale C if necessary
    if β != one(β)
        @sync for p in C.pids
            if iszero(β)
                @async remotecall_wait(() -> fill!(localpart(C), 0), p)
            else
                @async remotecall_wait(() -> rmul!(localpart(C), β), p)
            end
        end
    end

    # Update C
    @sync for i = 1:size(R, 1)
        for k = 1:size(C.pids, 2)
            p = C.pids[i,k]
            for j = 1:size(R, 2)
                rijk = R[i,j,k]
                @async remotecall_wait(d -> add!(localpart(d), fetch(rijk), α), p, C)
            end
        end
    end
    return C
end

LinearAlgebra.mul!(C::DMatrix, A::DMatrix, B::AbstractMatrix, α::Number = 1, β::Number = 0) = _matmatmul!(C, A, B, α, β, 'N')
LinearAlgebra.mul!(C::DMatrix, A::Adjoint{<:Number,<:DMatrix}, B::AbstractMatrix, α::Number = 1, β::Number = 0) = _matmatmul!(C, parent(A), B, α, β, 'C')
LinearAlgebra.mul!(C::DMatrix, A::Transpose{<:Number,<:DMatrix}, B::AbstractMatrix, α::Number = 1, β::Number = 0) = _matmatmul!(C, parent(A), B, α, β, 'T')

_matmul_op = (t,s) -> t*s + t*s

function Base.:*(A::DMatrix, x::AbstractVector)
    T = Base.promote_op(_matmul_op, eltype(A), eltype(x))
    y = DArray(I -> Array{T}(undef, map(length, I)), (size(A, 1),), procs(A)[:,1], (size(procs(A), 1),))
    return mul!(y, A, x)
end
function Base.:*(A::DMatrix, B::AbstractMatrix)
    T = Base.promote_op(_matmul_op, eltype(A), eltype(B))
    C = DArray(I -> Array{T}(undef, map(length, I)),
            (size(A, 1), size(B, 2)),
            procs(A)[:,1:min(size(procs(A), 2), size(procs(B), 2))],
            (size(procs(A), 1), min(size(procs(A), 2), size(procs(B), 2))))
    return mul!(C, A, B)
end

function Base.:*(adjA::Adjoint{<:Any,<:DMatrix}, x::AbstractVector)
    A = parent(adjA)
    T = Base.promote_op(_matmul_op, eltype(A), eltype(x))
    y = DArray(I -> Array{T}(undef, map(length, I)),
            (size(A, 2),),
            procs(A)[1,:],
            (size(procs(A), 2),))
    return mul!(y, adjA, x)
end
function Base.:*(adjA::Adjoint{<:Any,<:DMatrix}, B::AbstractMatrix)
    A = parent(adjA)
    T = Base.promote_op(_matmul_op, eltype(A), eltype(B))
    C = DArray(I -> Array{T}(undef, map(length, I)), (size(A, 2),
        size(B, 2)),
        procs(A)[1:min(size(procs(A), 1), size(procs(B), 2)),:],
        (size(procs(A), 2), min(size(procs(A), 1), size(procs(B), 2))))
    return mul!(C, adjA, B)
end

function Base.:*(trA::Transpose{<:Any,<:DMatrix}, x::AbstractVector)
    A = parent(trA)
    T = Base.promote_op(_matmul_op, eltype(A), eltype(x))
    y = DArray(I -> Array{T}(undef, map(length, I)),
            (size(A, 2),),
            procs(A)[1,:],
            (size(procs(A), 2),))
    return mul!(y, trA, x)
end
function Base.:*(trA::Transpose{<:Any,<:DMatrix}, B::AbstractMatrix)
    A = parent(trA)
    T = Base.promote_op(_matmul_op, eltype(A), eltype(B))
    C = DArray(I -> Array{T}(undef, map(length, I)), (size(A, 2),
        size(B, 2)),
        procs(A)[1:min(size(procs(A), 1), size(procs(B), 2)),:],
        (size(procs(A), 2), min(size(procs(A), 1), size(procs(B), 2))))
    return mul!(C, trA, B)
end
