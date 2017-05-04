"""
    DArray(init, dims, [procs, dist])

Construct a distributed array.

The parameter `init` is a function that accepts a tuple of index ranges.
This function should allocate a local chunk of the distributed array and initialize it for the specified indices.

`dims` is the overall size of the distributed array.

`procs` optionally specifies a vector of process IDs to use.
If unspecified, the array is distributed over all worker processes only. Typically, when running in distributed mode,
i.e., nprocs() > 1, this would mean that no chunk of the distributed array exists on the process hosting the
interactive julia prompt.

`dist` is an integer vector specifying how many chunks the distributed array should be divided into in each dimension.

For example, the `dfill` function that creates a distributed array and fills it with a value `v` is implemented as:

### Example
```jl
dfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)
```
"""
type DArray{T,N,A} <: AbstractArray{T,N}
    id::Tuple
    dims::NTuple{N,Int}
    pids::Array{Int,N}                          # pids[i]==p ⇒ processor p has piece i
    indexes::Array{NTuple{N,UnitRange{Int}},N}  # indexes held by piece i
    cuts::Vector{Vector{Int}}                   # cuts[d][i] = first index of chunk i in dimension d
    localpart::Nullable{A}

    release::Bool

    function DArray{T,N,A}(id, dims, pids, indexes, cuts, lp) where {T,N,A}
        # check invariants
        if dims != map(last, last(indexes))
            throw(ArgumentError("dimension of DArray (dim) and indexes do not match"))
        end
        release = (myid() == id[1])

        haskey(registry, id) && return registry[id]

        d = new(id, dims, pids, indexes, cuts, lp, release)
        if release
            push!(refs, id)
            registry[id] = d

#            println("Installing finalizer for : ", d.id, ", : ", object_id(d), ", isbits: ", isbits(d))
            finalizer(d, close)
        end
        d
    end

    DArray{T,N,A}() where {T,N,A} = new()
end

eltype{T}(::Type{DArray{T}}) = T
empty_localpart(T,N,A) = convert(A, Array{T}(ntuple(zero, N)))

const SubDArray{T,N,D<:DArray} = SubArray{T,N,D}
const SubOrDArray{T,N} = Union{DArray{T,N}, SubDArray{T,N}}

localtype{T,N,S}(::Type{DArray{T,N,S}}) = S
localtype{T,N,D}(::Type{SubDArray{T,N,D}}) = localtype(D)
localtype(A::SubOrDArray) = localtype(typeof(A))
localtype(A::AbstractArray) = typeof(A)

Base.hash(d::DArray, h::UInt) = Base.hash(d.id, h)

## core constructors ##

function DArray(id, init, dims, pids, idxs, cuts)
    r=Channel(1)
    @sync begin
        for i = 1:length(pids)
            @async begin
                local typA
                if isa(init, Function)
                    typA=remotecall_fetch(construct_localparts, pids[i], init, id, dims, pids, idxs, cuts)
                else
                    # constructing from an array of remote refs.
                    typA=remotecall_fetch(construct_localparts, pids[i], init[i], id, dims, pids, idxs, cuts)
                end
                !isready(r) && put!(r, typA)
            end
        end
    end

    A = take!(r)
    if myid() in pids
        d = registry[id]
    else
        T = eltype(A)
        N = length(dims)
        d = DArray{T,N,A}(id, dims, pids, idxs, cuts, empty_localpart(T,N,A))
    end
    d
end

function construct_localparts(init, id, dims, pids, idxs, cuts; T=nothing, A=nothing)
    localpart = isa(init, Function) ? init(idxs[localpartindex(pids)]) : fetch(init)
    if A == nothing
        A = typeof(localpart)
    end
    if T == nothing
        T = eltype(A)
    end
    N = length(dims)
    d = DArray{T,N,A}(id, dims, pids, idxs, cuts, localpart)
    registry[id] = d
    A
end

function ddata(;T::Type=Any, init::Function=I->nothing, pids=workers(), data::Vector=[])
    pids=sort(vec(pids))
    id = next_did()
    npids = length(pids)
    ldata = length(data)
    idxs, cuts = chunk_idxs([npids], [npids])

    if ldata > 0
        @assert rem(ldata,npids) == 0
        if ldata == npids
            T = eltype(data)
            s = DestinationSerializer(pididx->data[pididx], pids)
            init = I->localpart(s)
        else
            # call the standard distribute function
            return distribute(data)
        end
    end

    @sync for i = 1:length(pids)
        @async remotecall_fetch(construct_localparts, pids[i], init, id, (npids,), pids, idxs, cuts; T=T, A=T)
    end

    if myid() in pids
        d = registry[id]
    else
        d = DArray{T,1,T}(id, (npids,), pids, idxs, cuts, Nullable{T}())
    end
    d
end

function gather{T}(d::DArray{T,1,T})
    a=Array{T}(length(procs(d)))
    @sync for (i,p) in enumerate(procs(d))
        @async a[i] = remotecall_fetch(localpart, p, d)
    end
    a
end

function DArray(init, dims, procs, dist)
    np = prod(dist)
    procs = reshape(procs[1:np], ntuple(i->dist[i], length(dist)))
    idxs, cuts = chunk_idxs([dims...], dist)
    id = next_did()

    return DArray(id, init, dims, procs, idxs, cuts)
end

function DArray(init, dims, procs)
    if isempty(procs)
        throw(ArgumentError("no processors given"))
    end
    return DArray(init, dims, procs, defaultdist(dims, procs))
end
DArray(init, dims) = DArray(init, dims, workers()[1:min(nworkers(), maximum(dims))])

# Create a DArray from a collection of references
# The refs must have the same layout as the parts distributed.
# i.e.
#    size(refs) must specify the distribution of dimensions across processors
#    prod(size(refs)) must equal number of parts
# FIXME : Empty parts are currently not supported.
function DArray(refs)
    dimdist = size(refs)
    id = next_did()

    npids = [r.where for r in refs]
    nsizes = Array{Tuple}(dimdist)
    @sync for i in 1:length(refs)
        let i=i
            @async nsizes[i] = remotecall_fetch(sz_localpart_ref, npids[i], refs[i], id)
        end
    end

    nindexes = Array{NTuple{length(dimdist),UnitRange{Int}}}(dimdist...)

    for i in 1:length(nindexes)
        subidx = ind2sub(dimdist, i)
        nindexes[i] = ntuple(length(subidx)) do x
            idx_in_dim = subidx[x]
            startidx = 1
            for j in 1:(idx_in_dim-1)
                prevsubidx = ntuple(y -> y == x ? j : subidx[y], length(subidx))
                prevsize = nsizes[prevsubidx...]
                startidx += prevsize[x]
            end
            startidx:startidx+(nsizes[i][x])-1
        end
    end

    lastidxs = hcat([Int[last(idx_in_d)+1 for idx_in_d in idx] for idx in nindexes]...)
    ncuts = Array{Int,1}[unshift!(sort(unique(lastidxs[x,:])), 1) for x in 1:length(dimdist)]
    ndims = tuple([sort(unique(lastidxs[x,:]))[end]-1 for x in 1:length(dimdist)]...)

    DArray(id, refs, ndims, reshape(npids, dimdist), nindexes, ncuts)
end

macro DArray(ex0::Expr)
    if ex0.head !== :comprehension
        throw(ArgumentError("invalid @DArray syntax"))
    end
    ex = ex0.args[1]
    if ex.head !== :generator
        throw(ArgumentError("invalid @DArray syntax"))
    end
    ex.args[1] = esc(ex.args[1])
    ndim = length(ex.args) - 1
    ranges = map(r->esc(r.args[2]), ex.args[2:end])
    for d = 1:ndim
        var = ex.args[d+1].args[1]
        ex.args[d+1] = :( $(esc(var)) = ($(ranges[d]))[I[$d]] )
    end
    return :( DArray((I::Tuple{Vararg{UnitRange{Int}}})->($ex0),
                tuple($(map(r->:(length($r)), ranges)...))) )
end

# new DArray similar to an existing one
DArray(init, d::DArray) = DArray(next_did(), init, size(d), procs(d), d.indexes, d.cuts)

sz_localpart_ref(ref, id) = size(fetch(ref))

Base.similar(d::DArray, T::Type, dims::Dims) = DArray(I->Array{T}(map(length,I)), dims, procs(d))
Base.similar(d::DArray, T::Type) = similar(d, T, size(d))
Base.similar{T}(d::DArray{T}, dims::Dims) = similar(d, T, dims)
Base.similar{T}(d::DArray{T}) = similar(d, T, size(d))

Base.size(d::DArray) = d.dims

chunktype{T,N,A}(d::DArray{T,N,A}) = A

## chunk index utilities ##

# decide how to divide each dimension
# returns size of chunks array
function defaultdist(dims, pids)
    dims = [dims...]
    chunks = ones(Int, length(dims))
    np = length(pids)
    f = sort!(collect(keys(factor(np))), rev=true)
    k = 1
    while np > 1
        # repeatedly allocate largest factor to largest dim
        if np % f[k] != 0
            k += 1
            if k > length(f)
                break
            end
        end
        fac = f[k]
        (d, dno) = findmax(dims)
        # resolve ties to highest dim
        dno = last(find(dims .== d))
        if dims[dno] >= fac
            dims[dno] = div(dims[dno], fac)
            chunks[dno] *= fac
        end
        np = div(np, fac)
    end
    return chunks
end

# get array of start indexes for dividing sz into nc chunks
function defaultdist(sz::Int, nc::Int)
    if sz >= nc
        return round.(Int, linspace(1, sz+1, nc+1))
    else
        return [[1:(sz+1);], zeros(Int, nc-sz);]
    end
end

# compute indexes array for dividing dims into chunks
function chunk_idxs(dims, chunks)
    cuts = map(defaultdist, dims, chunks)
    n = length(dims)
    idxs = Array{NTuple{n,UnitRange{Int}}}(chunks...)
    for cidx in CartesianRange(tuple(chunks...))
        idxs[cidx.I...] = ntuple(i -> (cuts[i][cidx[i]]:cuts[i][cidx[i] + 1] - 1), n)
    end
    return (idxs, cuts)
end

function localpartindex(pids::Array{Int})
    mi = myid()
    for i = 1:length(pids)
        if pids[i] == mi
            return i
        end
    end
    return 0
end
localpartindex(d::DArray) = localpartindex(procs(d))

"""
    localpart(d::DArray)

Get the local piece of a distributed array.
Returns an empty array if no local part exists on the calling process.

d[:L], d[:l], d[:LP], d[:lp] are an alternative means to get localparts.
This syntaxt can also be used for assignment. For example,
`d[:L]=v` will assign `v` to the localpart of `d`.
"""
function localpart{T,N,A}(d::DArray{T,N,A})
    lpidx = localpartindex(d)
    if lpidx == 0
        return empty_localpart(T,N,A)::A
    end

    return get(registry[d.id].localpart)::A
end

localpart(d::DArray, localidx...) = localpart(d)[localidx...]

# shortcut to set/get localparts of a distributed object
function Base.getindex(d::DArray, s::Symbol)
    @assert s in [:L, :l, :LP, :lp]
    return localpart(d)
end

function Base.setindex!{T,N,A}(d::DArray{T,N,A}, new_lp::A, s::Symbol)
    @assert s in [:L, :l, :LP, :lp]
    d.localpart = new_lp
    new_lp
end


# fetch localpart of d at pids[i]
fetch{T,N,A}(d::DArray{T,N,A}, i) = remotecall_fetch(localpart, d.pids[i], d)

"""
    localindexes(d)

A tuple describing the indexes owned by the local process.
Returns a tuple with empty ranges if no local part exists on the calling process.
"""
function localindexes(d::DArray)
    lpidx = localpartindex(d)
    if lpidx == 0
        return ntuple(i -> 1:0, ndims(d))
    end
    return d.indexes[lpidx]
end

# find which piece holds index (I...)
locate(d::DArray, I::Int...) =
    ntuple(i -> searchsortedlast(d.cuts[i], I[i]), ndims(d))

chunk{T,N,A}(d::DArray{T,N,A}, i...) = remotecall_fetch(localpart, d.pids[i...], d)::A

## convenience constructors ##

"""
     dzeros(dims, ...)

Construct a distributed array of zeros.
Trailing arguments are the same as those accepted by `DArray`.
"""
dzeros(dims::Dims, args...) = DArray(I->zeros(map(length,I)), dims, args...)
dzeros{T}(::Type{T}, dims::Dims, args...) = DArray(I->zeros(T,map(length,I)), dims, args...)
dzeros{T}(::Type{T}, d1::Integer, drest::Integer...) = dzeros(T, convert(Dims, tuple(d1, drest...)))
dzeros(d1::Integer, drest::Integer...) = dzeros(Float64, convert(Dims, tuple(d1, drest...)))
dzeros(d::Dims) = dzeros(Float64, d)


"""
    dones(dims, ...)

Construct a distributed array of ones.
Trailing arguments are the same as those accepted by `DArray`.
"""
dones(dims::Dims, args...) = DArray(I->ones(map(length,I)), dims, args...)
dones{T}(::Type{T}, dims::Dims, args...) = DArray(I->ones(T,map(length,I)), dims, args...)
dones{T}(::Type{T}, d1::Integer, drest::Integer...) = dones(T, convert(Dims, tuple(d1, drest...)))
dones(d1::Integer, drest::Integer...) = dones(Float64, convert(Dims, tuple(d1, drest...)))
dones(d::Dims) = dones(Float64, d)

"""
     dfill(x, dims, ...)

Construct a distributed array filled with value `x`.
Trailing arguments are the same as those accepted by `DArray`.
"""
dfill(v, dims::Dims, args...) = DArray(I->fill(v, map(length,I)), dims, args...)
dfill(v, d1::Integer, drest::Integer...) = dfill(v, convert(Dims, tuple(d1, drest...)))

"""
     drand(dims, ...)

Construct a distributed uniform random array.
Trailing arguments are the same as those accepted by `DArray`.
"""
drand(r, dims::Dims, args...) = DArray(I -> rand(r, map(length,I)), dims, args...)
drand(r, d1::Integer, drest::Integer...) = drand(r, convert(Dims, tuple(d1, drest...)))
drand(d1::Integer, drest::Integer...) = drand(Float64, convert(Dims, tuple(d1, drest...)))
drand(d::Dims, args...)  = drand(Float64, d, args...)

"""
     drandn(dims, ...)

Construct a distributed normal random array.
Trailing arguments are the same as those accepted by `DArray`.
"""
drandn(dims::Dims, args...) = DArray(I->randn(map(length,I)), dims, args...)
drandn(d1::Integer, drest::Integer...) = drandn(convert(Dims, tuple(d1, drest...)))

## conversions ##

"""
     distribute(A[; procs, dist])

Convert a local array to distributed.

`procs` optionally specifies an array of process IDs to use. (defaults to all workers)
`dist` optionally specifies a vector or tuple of the number of partitions in each dimension
"""
function distribute(A::AbstractArray;
    procs = workers()[1:min(nworkers(), maximum(size(A)))],
    dist = defaultdist(size(A), procs))
    np = prod(dist)
    procs_used = procs[1:np]
    idxs, _ = chunk_idxs([size(A)...], dist)

    s = verified_destination_serializer(reshape(procs_used, size(idxs)), size(idxs)) do pididx
        A[idxs[pididx]...]
    end
    return DArray(I->localpart(s), size(A), procs_used, dist)
end

"""
    distribute(A, DA)

Distribute a local array `A` like the distributed array `DA`.

"""
function distribute(A::AbstractArray, DA::DArray)
    size(DA) == size(A) || throw(DimensionMismatch("Distributed array has size $(size(DA)) but array has $(size(A))"))

    s = verified_destination_serializer(procs(DA), size(DA.indexes)) do pididx
        A[DA.indexes[pididx]...]
    end
    return DArray(I->localpart(s), DA)
end

Base.convert{T,N,S<:AbstractArray}(::Type{DArray{T,N,S}}, A::S) = distribute(convert(AbstractArray{T,N}, A))

Base.convert{S,T,N}(::Type{Array{S,N}}, d::DArray{T,N}) = begin
    a = Array{S}(size(d))
    @sync begin
        for i = 1:length(d.pids)
            @async a[d.indexes[i]...] = chunk(d, i)
        end
    end
    return a
end

Base.convert{S,T,N}(::Type{Array{S,N}}, s::SubDArray{T,N}) = begin
    I = s.indexes
    d = s.parent
    if isa(I,Tuple{Vararg{UnitRange{Int}}}) && S<:T && T<:S
        l = locate(d, map(first, I)...)
        if isequal(d.indexes[l...], I)
            # SubDArray corresponds to a chunk
            return chunk(d, l...)
        end
    end
    a = Array{S}(size(s))
    a[[1:size(a,i) for i=1:N]...] = s
    return a
end

function Base.convert{T,N}(::Type{DArray}, SD::SubArray{T,N})
    D = SD.parent
    DArray(size(SD), procs(D)) do I
        TR = typeof(SD.indexes[1])
        lindices = Array{TR}(0)
        for (i,r) in zip(I, SD.indexes)
            st = step(r)
            lrstart = first(r) + st*(first(i)-1)
            lrend = first(r) + st*(last(i)-1)
            if TR <: UnitRange
                push!(lindices, lrstart:lrend)
            else
                push!(lindices, lrstart:st:lrend)
            end
        end
        convert(Array, D[lindices...])
    end
end

Base.reshape{T,S<:Array}(A::DArray{T,1,S}, d::Dims) = begin
    if prod(d) != length(A)
        throw(DimensionMismatch("dimensions must be consistent with array size"))
    end
    return DArray(d) do I
        sz = map(length,I)
        d1offs = first(I[1])
        nd = length(I)

        B = Array{T}(sz)
        nr = size(B,1)
        sztail = size(B)[2:end]

        for i=1:div(length(B),nr)
            i2 = ind2sub(sztail, i)
            globalidx = [ I[j][i2[j-1]] for j=2:nd ]

            a = sub2ind(d, d1offs, globalidx...)

            B[:,i] = A[a:(a+nr-1)]
        end
        B
    end
end

## indexing ##

getlocalindex(d::DArray, idx...) = localpart(d)[idx...]
function getindex_tuple{T}(d::DArray{T}, I::Tuple{Vararg{Int}})
    chidx = locate(d, I...)
    idxs = d.indexes[chidx...]
    localidx = ntuple(i -> (I[i] - first(idxs[i]) + 1), ndims(d))
    pid = d.pids[chidx...]
    return remotecall_fetch(getlocalindex, pid, d, localidx...)::T
end

Base.getindex(d::DArray, i::Int) = getindex_tuple(d, ind2sub(size(d), i))
Base.getindex(d::DArray, i::Int...) = getindex_tuple(d, i)

Base.getindex(d::DArray) = d[1]
Base.getindex(d::DArray, I::Union{Int,UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...) = view(d, I...)


Base.copy!(dest::SubOrDArray, src::SubOrDArray) = begin
    asyncmap(procs(dest)) do p
        remotecall_fetch(p) do
            localpart(dest)[:] = src[localindexes(dest)...]
        end
    end
    return dest
end

# local copies are obtained by convert(Array, ) or assigning from
# a SubDArray to a local Array.

function Base.setindex!(a::Array, d::DArray,
        I::Union{UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...)
    n = length(I)
    @sync for i = 1:length(d.pids)
        K = d.indexes[i]
        @async a[[I[j][K[j]] for j=1:n]...] = chunk(d, i)
    end
    return a
end

# We also want to optimize setindex! with a SubDArray source, but this is hard
# and only works on 0.5.

# Similar to Base.indexin, but just create a logical mask. Note that this
# must return a logical mask in order to support merging multiple masks
# together into one linear index since we need to know how many elements to
# skip at the end. In many cases range intersection would be much faster
# than generating a logical mask, but that loses the endpoint information.
indexin_mask(a, b::Number) = a .== b
indexin_mask(a, r::Range{Int}) = [i in r for i in a]
indexin_mask(a, b::AbstractArray{Int}) = indexin_mask(a, IntSet(b))
indexin_mask(a, b::AbstractArray) = indexin_mask(a, Set(b))
indexin_mask(a, b) = [i in b for i in a]

import Base: tail
# Given a tuple of indices and a tuple of masks, restrict the indices to the
# valid regions. This is, effectively, reversing Base.setindex_shape_check.
# We can't just use indexing into MergedIndices here because getindex is much
# pickier about singleton dimensions than setindex! is.
restrict_indices(::Tuple{}, ::Tuple{}) = ()
function restrict_indices(a::Tuple{Any, Vararg{Any}}, b::Tuple{Any, Vararg{Any}})
    if (length(a[1]) == length(b[1]) == 1) || (length(a[1]) > 1 && length(b[1]) > 1)
        (vec(a[1])[vec(b[1])], restrict_indices(tail(a), tail(b))...)
    elseif length(a[1]) == 1
        (a[1], restrict_indices(tail(a), b))
    elseif length(b[1]) == 1 && b[1][1]
        restrict_indices(a, tail(b))
    else
        throw(DimensionMismatch("this should be caught by setindex_shape_check; please submit an issue"))
    end
end
# The final indices are funky - they're allowed to accumulate together.
# An easy (albeit very inefficient) fix for too many masks is to use the
# outer product to merge them. But we can do that lazily with a custom type:
function restrict_indices(a::Tuple{Any}, b::Tuple{Any, Any, Vararg{Any}})
    (vec(a[1])[vec(ProductIndices(b, map(length, b)))],)
end
# But too many indices is much harder; this requires merging the indices
# in `a` before applying the final mask in `b`.
function restrict_indices(a::Tuple{Any, Any, Vararg{Any}}, b::Tuple{Any})
    if length(a[1]) == 1
        (a[1], restrict_indices(tail(a), b))
    else
        # When one mask spans multiple indices, we need to merge the indices
        # together. At this point, we can just use indexing to merge them since
        # there's no longer special handling of singleton dimensions
        (view(MergedIndices(a, map(length, a)), b[1]),)
    end
end

immutable ProductIndices{I,N} <: AbstractArray{Bool, N}
    indices::I
    sz::NTuple{N,Int}
end
Base.size(P::ProductIndices) = P.sz
# This gets passed to map to avoid breaking propagation of inbounds
Base.@propagate_inbounds propagate_getindex(A, I...) = A[I...]
Base.@propagate_inbounds Base.getindex{_,N}(P::ProductIndices{_,N}, I::Vararg{Int, N}) =
    Bool((&)(map(propagate_getindex, P.indices, I)...))

immutable MergedIndices{I,N} <: AbstractArray{CartesianIndex{N}, N}
    indices::I
    sz::NTuple{N,Int}
end
Base.size(M::MergedIndices) = M.sz
Base.@propagate_inbounds Base.getindex{_,N}(M::MergedIndices{_,N}, I::Vararg{Int, N}) =
    CartesianIndex(map(propagate_getindex, M.indices, I))
# Additionally, we optimize bounds checking when using MergedIndices as an
# array index since checking, e.g., A[1:500, 1:500] is *way* faster than
# checking an array of 500^2 elements of CartesianIndex{2}. This optimization
# also applies to reshapes of MergedIndices since the outer shape of the
# container doesn't affect the index elements themselves. We can go even
# farther and say that even restricted views of MergedIndices must be valid
# over the entire array. This is overly strict in general, but in this
# use-case all the merged indices must be valid at some point, so it's ok.
const ReshapedMergedIndices{T,N,M<:MergedIndices} = Base.ReshapedArray{T,N,M}
const SubMergedIndices{T,N,M<:Union{MergedIndices, ReshapedMergedIndices}} = SubArray{T,N,M}
const MergedIndicesOrSub = Union{MergedIndices, ReshapedMergedIndices, SubMergedIndices}
import Base: checkbounds_indices
@inline checkbounds_indices(::Type{Bool}, inds::Tuple{}, I::Tuple{MergedIndicesOrSub,Vararg{Any}}) =
    checkbounds_indices(Bool, inds, (parent(parent(I[1])).indices..., tail(I)...))
@inline checkbounds_indices(::Type{Bool}, inds::Tuple{Any}, I::Tuple{MergedIndicesOrSub,Vararg{Any}}) =
    checkbounds_indices(Bool, inds, (parent(parent(I[1])).indices..., tail(I)...))
@inline checkbounds_indices(::Type{Bool}, inds::Tuple, I::Tuple{MergedIndicesOrSub,Vararg{Any}}) =
    checkbounds_indices(Bool, inds, (parent(parent(I[1])).indices..., tail(I)...))

# The tricky thing here is that we want to optimize the accesses into the
# distributed array, but in doing so, we lose track of which indices in I we
# should be using.
#
# I’ve come to the conclusion that the function is utterly insane.
# There are *6* flavors of indices with four different reference points:
# 1. Find the indices of each portion of the DArray.
# 2. Find the valid subset of indices for the SubArray into that portion.
# 3. Find the portion of the `I` indices that should be used when you access the
#    `K` indices in the subarray.  This guy is nasty.  It’s totally backwards
#    from all other arrays, wherein we simply iterate over the source array’s
#    elements.  You need to *both* know which elements in `J` were skipped
#    (`indexin_mask`) and which dimensions should match up (`restrict_indices`)
# 4. If `K` doesn’t correspond to an entire chunk, reinterpret `K` in terms of
#    the local portion of the source array
function Base.setindex!(a::Array, s::SubDArray,
        I::Union{UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...)
    Inew = Base.to_indices(a, I)
    Base.setindex_shape_check(s, Base.index_lengths(Inew...)...)
    n = length(Inew)
    d = s.parent
    J = Base.to_indices(d, s.indexes)
    @sync for i = 1:length(d.pids)
        K_c = d.indexes[i]
        K = map(intersect, J, K_c)
        if !any(isempty, K)
            K_mask = map(indexin_mask, J, K_c)
            idxs = restrict_indices(Inew, K_mask)
            if isequal(K, K_c)
                # whole chunk
                @async a[idxs...] = chunk(d, i)
            else
                # partial chunk
                @async a[idxs...] =
                    remotecall_fetch(d.pids[i]) do
                        view(localpart(d), [K[j]-first(K_c[j])+1 for j=1:length(J)]...)
                    end
            end
        end
    end
    return a
end

Base.fill!(A::DArray, x) = begin
    @sync for p in procs(A)
        @async remotecall_fetch((A,x)->(fill!(localpart(A), x); nothing), p, A, x)
    end
    return A
end
