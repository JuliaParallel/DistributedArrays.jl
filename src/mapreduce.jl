## higher-order functions ##

import Base: +, -, div, mod, rem, &, |, xor
import SparseArrays: nnz

Base.map(f, d0::DArray, ds::AbstractArray...) = broadcast(f, d0, ds...)

function Base.map!(f::F, dest::DArray, src::DArray{<:Any,<:Any,A}) where {F,A}
    asyncmap(procs(dest)) do p
        remotecall_fetch(p) do
            map!(f, localpart(dest), A(view(src, localindices(dest)...)))
            return nothing
        end
    end
    return dest
end

function Base.reduce(f, d::DArray)
    results = asyncmap(procs(d)) do p
        remotecall_fetch(p) do
            return reduce(f, localpart(d))
        end
    end
    reduce(f, results)
end

function Base._mapreduce(f, op, ::IndexCartesian, d::DArray)
    results = asyncmap(procs(d)) do p
        remotecall_fetch((_f,_op,_d)->mapreduce(_f, _op, localpart(_d)), p, f, op, d)
    end

    reduce(op, results)
end
Base._mapreduce(f, op, ::IndexCartesian, d::SubDArray) = Base._mapreduce(f, op, IndexCartesian(), DArray(d))
# Base.mapreduce(f, opt::Union{typeof(|), typeof(&)}, d::DArray) = _mapreduce(f, opt, d)
# Base.mapreduce(f, opt::Function, d::DArray) = _mapreduce(f, opt, d)
# Base.mapreduce(f, opt, d::DArray) = _mapreduce(f, opt, d)

# mapreducedim
function Base.reducedim_initarray(A::DArray, region, v0, ::Type{R}) where {R}
    # Store reduction on lowest pids
    pids = A.pids[ntuple(i -> i in region ? (1:1) : (:), ndims(A))...]
    chunks = similar(pids, Future)
    @sync for i in eachindex(pids)
        @async chunks[i...] = remotecall_wait(() -> Base.reducedim_initarray(localpart(A), region, v0, R), pids[i...])
    end
    return DArray(chunks)
end
Base.reducedim_initarray(A::DArray, region, v0::T) where {T} = Base.reducedim_initarray(A, region, v0, T)

# Compute mapreducedim of each localpart and store the result in a new DArray
function mapreducedim_within(f, op, A::DArray, region)
    arraysize = [size(A)...]
    gridsize = [size(A.indices)...]
    arraysize[[region...]] = gridsize[[region...]]
    indx = similar(A.indices)

    for i in CartesianIndices(indx)
        indx[i] = ntuple(j -> j in region ? (i.I[j]:i.I[j]) : A.indices[i][j], ndims(A))
    end
    cuts = [i in region ? collect(1:arraysize[i] + 1) : A.cuts[i] for i in 1:ndims(A)]
    return DArray(next_did(), I -> mapreduce(f, op, localpart(A), dims=region),
        tuple(arraysize...), procs(A), indx, cuts)
end

# Compute mapreducedim accros the processes. This should be done after mapreducedim
# has been run on each localpart with mapreducedim_within. Eventually, we might
# want to write mapreducedim_between! as a binary reduction.
function mapreducedim_between!(f, op, R::DArray, A::DArray, region)
    asyncmap(procs(R)) do p
        remotecall_fetch(p, f, op, R, A, region) do f, op, R, A, region
            localind = [r for r = localindices(A)]
            localind[[region...]] = [1:n for n = size(A)[[region...]]]
            B = convert(Array, A[localind...])
            Base.mapreducedim!(f, op, localpart(R), B)
            nothing
        end
    end
    return R
end

function Base.mapreducedim!(f, op, R::DArray, A::DArray)
    lsize = Base.check_reducedims(R,A)
    if isempty(A)
        return copy(R)
    end
    region = tuple(collect(1:ndims(A))[[size(R)...] .!= [size(A)...]]...)
    if isempty(region)
        return copyto!(R, A)
    end
    B = mapreducedim_within(f, op, A, region)
    return mapreducedim_between!(identity, op, R, B, region)
end

## Some special cases
function Base._all(f, A::DArray, ::Colon)
    B = asyncmap(procs(A)) do p
        remotecall_fetch(p) do
            all(f, localpart(A))
        end
    end
    return all(B)
end

function Base._any(f, A::DArray, ::Colon)
    B = asyncmap(procs(A)) do p
        remotecall_fetch(p) do
            any(f, localpart(A))
        end
    end
    return any(B)
end

function Base.count(f, A::DArray)
    B = asyncmap(procs(A)) do p
        remotecall_fetch(p) do
            count(f, localpart(A))
        end
    end
    return sum(B)
end

function nnz(A::DArray)
    B = asyncmap(A.pids) do p
        remotecall_fetch(nnz∘localpart, p, A)
    end
    return reduce(+, B)
end

function Base.extrema(d::DArray)
    r = asyncmap(procs(d)) do p
        remotecall_fetch(p) do
            extrema(localpart(d))
        end
    end
    return reduce((t,s) -> (min(t[1], s[1]), max(t[2], s[2])), r)
end

Statistics._mean(A::DArray, region) = sum(A, dims = region) ./ prod((size(A, i) for i in region))

# Unary vector functions
(-)(D::DArray) = map(-, D)


map_localparts(f::Callable, d::DArray) = DArray(i->f(localpart(d)), d)
map_localparts(f::Callable, d1::DArray, d2::DArray) = DArray(d1) do I
    f(localpart(d1), localpart(d2))
end

function map_localparts(f::Callable, DA::DArray, A::Array)
    s = verified_destination_serializer(procs(DA), size(DA.indices)) do pididx
        A[DA.indices[pididx]...]
    end
    DArray(DA) do I
        f(localpart(DA), localpart(s))
    end
end

function map_localparts(f::Callable, A::Array, DA::DArray)
    s = verified_destination_serializer(procs(DA), size(DA.indices)) do pididx
        A[DA.indices[pididx]...]
    end
    DArray(DA) do I
        f(localpart(s), localpart(DA))
    end
end

function map_localparts!(f::Callable, d::DArray)
    asyncmap(procs(d)) do p
        remotecall_fetch((f,d)->(f(localpart(d)); nothing), p, f, d)
    end
    return d
end

# Here we assume all the DArrays have
# the same size and distribution
map_localparts(f::Callable, As::DArray...) = DArray(I->f(map(localpart, As)...), As[1])


function samedist(A::DArray, B::DArray)
    (size(A) == size(B)) || throw(DimensionMismatch())
    if (procs(A) != procs(B)) || (A.cuts != B.cuts)
        B = DArray(x->B[x...], A)
    end
    B
end

for f in (:+, :-, :div, :mod, :rem, :&, :|, :xor)
    @eval begin
        function ($f)(A::DArray{T}, B::DArray{T}) where T
            B = samedist(A, B)
            map_localparts($f, A, B)
        end
        ($f)(A::DArray{T}, B::Array{T}) where {T} = map_localparts($f, A, B)
        ($f)(A::Array{T}, B::DArray{T}) where {T} = map_localparts($f, A, B)
    end
end

function Base.mapslices(f, D::DArray{T,N,A}; dims) where {T,N,A}
    if !(dims isa AbstractVector)
        dims = [dims...]
    end
    if !all(t -> t == 1, size(D.indices)[dims])
        p = ones(Int, ndims(D))
        nondims = filter(t -> !(t in dims), 1:ndims(D))
        p[nondims] = defaultdist([size(D)...][[nondims...]], procs(D))
        DD = DArray(size(D), procs(D), p) do I
            return convert(A, D[I...])
        end
        return mapslices(f, DD, dims=dims)
    end

    refs = Future[remotecall((x,y,z)->mapslices(x,localpart(y),dims=z), p, f, D, dims) for p in procs(D)]

    DArray(reshape(refs, size(procs(D))))
end

function _ppeval(f, A...; dim = map(ndims, A))
    if length(dim) != length(A)
        throw(ArgumentError("dim argument has wrong length. length(dim) = $(length(dim)) but should be $(length(A))"))
    end
    narg = length(A)
    dimlength = size(A[1], dim[1])
    for i = 2:narg
        if dim[i] > 0 && dimlength != size(A[i], dim[i])
            throw(ArgumentError("lengths of broadcast dimensions must be the same. size(A[1], $(dim[1])) = $dimlength but size(A[$i], $(dim[i])) = $(size(A[i], dim[i]))"))
        end
    end
    dims = []
    idx  = []
    args = []
    for i = 1:narg
        push!(dims, ndims(A[i]))
        push!(idx, Any[Colon() for d in 1:dims[i]])
        if dim[i] > 0
            idx[i][dim[i]] = 1
            push!(args, view(A[i], idx[i]...))
        else
            push!(args, A[i])
        end
    end
    R1 = f(args...)
    ridx = Any[1:size(R1, d) for d in 1:ndims(R1)]
    push!(ridx, 1)
    Rsize = map(last, ridx)
    Rsize[end] = dimlength
    R = Array{eltype(R1)}(undef, Rsize...)

    for i = 1:dimlength
        for j = 1:narg
            if dim[j] > 0
                idx[j][dim[j]] = i
                args[j] = view(A[j], idx[j]...)
            else
                args[j] = A[j]
            end
        end
        ridx[end] = i
        R[ridx...] = f(args...)
    end

    return R
end

"""
     ppeval(f, D...; dim::NTuple)

Evaluates the callable argument `f` on slices of the elements of the `D` tuple.

#### Arguments
`f` can be any callable object that accepts sliced or broadcasted elements of `D`.
The result returned from `f` must be either an array or a scalar.

`D` has any number of elements and the elements can have any type. If an element
of `D` is a distributed array along the dimension specified by `dim`. If an
element of `D` is not distributed, the element is by default broadcasted and
applied on all evaluations of `f`.

`dim` is a tuple of integers specifying the dimension over which the elements
of `D` is slices. The length of the tuple must therefore be the same as the
number of arguments `D`. By default distributed arrays are slides along the
last dimension. If the value is less than or equal to zero the element are
broadcasted to all evaluations of `f`.

#### Result
`ppeval` returns a distributed array of dimension `p+1` where the first `p`
sizes correspond to the sizes of return values of `f`. The last dimension of
the return array from `ppeval` has the same length as the dimension over which
the input arrays are sliced.

#### Examples
```jl
addprocs(Sys.CPU_THREADS)

using DistributedArrays

A = drandn((10, 10, Sys.CPU_THREADS), workers(), [1, 1, Sys.CPU_THREADS])

ppeval(eigvals, A)

ppeval(eigvals, A, randn(10,10)) # broadcasting second argument

B = drandn((10, Sys.CPU_THREADS), workers(), [1, Sys.CPU_THREADS])

ppeval(*, A, B)
```
"""
function ppeval(f, D...; dim::NTuple = map(t -> isa(t, DArray) ? ndims(t) : 0, D))
    #Ensure that the complete DArray is available on the specified dims on all processors
    for i = 1:length(D)
        if isa(D[i], DArray)
            for idxs in D[i].indices
                for d in setdiff(1:ndims(D[i]), dim[i])
                    if length(idxs[d]) != size(D[i], d)
                        throw(DimensionMismatch(string("dimension $d is distributed. ",
                            "ppeval requires dimension $d to be completely available on all processors.")))
                    end
                end
            end
        end
    end

    refs = Future[remotecall((x, y, z) -> _ppeval(x, map(localpart, y)...; dim = z), p, f, D, dim) for p in procs(D[1])]

    # The array of Futures has to be reshaped for the DArray constructor to work correctly.
    # This requires a fetch and the DArray is also fetching so it might be better to modify
    # the DArray constructor.
    sd = [size(D[1].pids)...]
    nd = remotecall_fetch((r)->ndims(fetch(r)), refs[1].where, refs[1])
    DArray(reshape(refs, tuple([sd[1:nd - 1]; sd[end]]...)))
end
