function Base.ctranspose{T}(D::DArray{T,2})
    DArray(reverse(size(D)), procs(D)) do I
        lp = Array(T, map(length, I))
        rp = convert(Array, D[reverse(I)...])
        ctranspose!(lp, rp)
    end
end

function Base.transpose{T}(D::DArray{T,2})
    DArray(reverse(size(D)), procs(D)) do I
        lp = Array(T, map(length, I))
        rp = convert(Array, D[reverse(I)...])
        transpose!(lp, rp)
    end
end

typealias DVector{T,A} DArray{T,1,A}
typealias DMatrix{T,A} DArray{T,2,A}

# Level 1

function axpy!(α, x::DVector, y::DVector)
    if length(x) != length(y)
        throw(DimensionMismatch("vectors must have same length"))
    end
    @sync for p in procs(y)
        @async remotecall_fetch(() -> (Base.axpy!(α, localpart(x), localpart(y)); nothing), p)
    end
    return y
end

function dot(x::DVector, y::DVector)
    if length(x) != length(y)
        throw(DimensionMismatch(""))
    end
    if (procs(x) != procs(y)) || (x.cuts != y.cuts)
        throw(ArgumentError("vectors don't have the same distribution. Not handled for efficiency reasons."))
    end

    results=Any[]
    @sync begin
        for i = eachindex(x.pids)
            @async push!(results, remotecall_fetch((x, y, i) -> dot(localpart(x), fetch(y, i)), x.pids[i], x, y, i))
        end
    end
    return reduce(+, results)
end

function norm(x::DVector, p::Real = 2)
    results = []
    @sync begin
        for pp in procs(x)
            @async push!(results, remotecall_fetch(() -> norm(localpart(x), p), pp))
        end
    end
    return norm(results, p)
end

Base.scale!(A::DArray, x::Number) = begin
    @sync for p in procs(A)
        @async remotecall_fetch((A,x)->(scale!(localpart(A), x); nothing), p, A, x)
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

function A_mul_B!(α::Number, A::DMatrix, x::AbstractVector, β::Number, y::DVector)

    # error checks
    if size(A, 2) != length(x)
        throw(DimensionMismatch(""))
    end
    if y.cuts[1] != A.cuts[1]
        throw(ArgumentError("cuts of output vector must match cuts of first dimension of matrix"))
    end

    # Multiply on each tile of A
    R = Array(Future, size(A.pids)...)
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
        @sync for p in y.pids
            if β != zero(β)
                @async remotecall_fetch(y -> (scale!(localpart(y), β); nothing), p, y)
            else
                @async remotecall_fetch(y -> (fill!(localpart(y), 0); nothing), p, y)
            end
        end
    end

    # Update y
    @sync for i = 1:size(R, 1)
        p = y.pids[i]
        for j = 1:size(R, 2)
            rij = R[i,j]
            @async remotecall_fetch(() -> (add!(localpart(y), fetch(rij), α); nothing), p)
        end
    end

    return y
end

function Ac_mul_B!(α::Number, A::DMatrix, x::AbstractVector, β::Number, y::DVector)

    # error checks
    if size(A, 1) != length(x)
        throw(DimensionMismatch(""))
    end
    if y.cuts[1] != A.cuts[2]
        throw(ArgumentError("cuts of output vector must match cuts of second dimension of matrix"))
    end

    # Multiply on each tile of A
    R = Array(Future, reverse(size(A.pids))...)
    for j = 1:size(A.pids, 1)
        xj = x[A.cuts[1][j]:A.cuts[1][j + 1] - 1]
        for i = 1:size(A.pids, 2)
            R[i,j] = remotecall(() -> localpart(A)'*convert(localtype(x), xj), procs(A)[j,i])
        end
    end

    # Scale y if necessary
    if β != one(β)
        @sync for p in y.pids
            if β != zero(β)
                @async remotecall_fetch(() -> (scale!(localpart(y), β); nothing), p)
            else
                @async remotecall_fetch(() -> (fill!(localpart(y), 0); nothing), p)
            end
        end
    end

    # Update y
    @sync for i = 1:size(R, 1)
        p = y.pids[i]
        for j = 1:size(R, 2)
            rij = R[i,j]
            @async remotecall_fetch(() -> (add!(localpart(y), fetch(rij), α); nothing), p)
        end
    end
    return y
end


# Level 3
function _matmatmul!(α::Number, A::DMatrix, B::AbstractMatrix, β::Number, C::DMatrix, tA)
    # error checks
    Ad1, Ad2 = (tA == 'N') ? (1,2) : (2,1)
    mA, nA = size(A, Ad1, Ad2)
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
        R = Array(Future, size(procs(A))..., size(procs(C), 2))
    else
        R = Array(Future, reverse(size(procs(A)))..., size(procs(C), 2))
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
                        return localpart(A).'*convert(localtype(B), Bjk)
                    elseif tA == 'C'
                        return localpart(A)'*convert(localtype(B), Bjk)
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
            if β != zero(β)
                @async remotecall_fetch(() -> (scale!(localpart(C), β); nothing), p)
            else
                @async remotecall_fetch(() -> (fill!(localpart(C), 0); nothing), p)
            end
        end
    end

    # Update C
    @sync for i = 1:size(R, 1)
        for k = 1:size(C.pids, 2)
            p = C.pids[i,k]
            for j = 1:size(R, 2)
                rijk = R[i,j,k]
                @async remotecall_fetch(d -> (add!(localpart(d), fetch(rijk), α); nothing), p, C)
            end
        end
    end
    return C
end

A_mul_B!(α::Number, A::DMatrix, B::AbstractMatrix, β::Number, C::DMatrix) = _matmatmul!(α, A, B, β, C, 'N')
Ac_mul_B!(α::Number, A::DMatrix, B::AbstractMatrix, β::Number, C::DMatrix) = _matmatmul!(α, A, B, β, C, 'C')
At_mul_B!(α::Number, A::DMatrix, B::AbstractMatrix, β::Number, C::DMatrix) = _matmatmul!(α, A, B, β, C, 'T')
At_mul_B!(C::DMatrix, A::DMatrix, B::AbstractMatrix) = At_mul_B!(one(eltype(C)), A, B, zero(eltype(C)), C)

function (*)(A::DMatrix, x::AbstractVector)
    T = promote_type(Base.LinAlg.arithtype(eltype(A)), Base.LinAlg.arithtype(eltype(x)))
    y = DArray(I -> Array(T, map(length, I)), (size(A, 1),), procs(A)[:,1], (size(procs(A), 1),))
    return A_mul_B!(one(T), A, x, zero(T), y)
end
function (*)(A::DMatrix, B::AbstractMatrix)
    T = promote_type(Base.LinAlg.arithtype(eltype(A)), Base.LinAlg.arithtype(eltype(B)))
    C = DArray(I -> Array(T, map(length, I)), (size(A, 1), size(B, 2)), procs(A)[:,1:min(size(procs(A), 2), size(procs(B), 2))], (size(procs(A), 1), min(size(procs(A), 2), size(procs(B), 2))))
    return A_mul_B!(one(T), A, B, zero(T), C)
end

function Ac_mul_B(A::DMatrix, x::AbstractVector)
    T = promote_type(Base.LinAlg.arithtype(eltype(A)), Base.LinAlg.arithtype(eltype(x)))
    y = DArray(I -> Array(T, map(length, I)), (size(A, 2),), procs(A)[1,:], (size(procs(A), 2),))
    return Ac_mul_B!(one(T), A, x, zero(T), y)
end
function Ac_mul_B(A::DMatrix, B::AbstractMatrix)
    T = promote_type(Base.LinAlg.arithtype(eltype(A)), Base.LinAlg.arithtype(eltype(B)))
    C = DArray(I -> Array(T, map(length, I)), (size(A, 2), size(B, 2)), procs(A)[1:min(size(procs(A), 1), size(procs(B), 2)),:], (size(procs(A), 2), min(size(procs(A), 1), size(procs(B), 2))))
    return Ac_mul_B!(one(T), A, B, zero(T), C)
end
