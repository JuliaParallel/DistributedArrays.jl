const registry=Dict{Tuple, Any}()
const refs=Set()  # Collection of darray identities created on this node

let DID::Int = 1
    global next_did
    next_did() = (id = DID; DID += 1; (myid(), id))
end

"""
    next_did()

Produces an incrementing ID that will be used for Distributeds.
"""
next_did

"""
    Distributed(init, dims, [procs, dist])

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
dfill(v, args...) = Distributed(I->fill(v, map(length,I)), args...)
```
"""
type Distributed{T,N,A,DistType} <: AbstractArray{T,N}
    id::Tuple
    dims::NTuple{N,Int}
    pids::Array{Int,N}                          # pids[i]==p ⇒ processor p has piece i
    indexes::Array{NTuple{N,UnitRange{Int}},N}  # indexes held by piece i
    cuts::Vector{Vector{Int}}                   # cuts[d][i] = first index of chunk i in dimension d

    release::Bool
    v::Nullable{A}                              # Always refers to the localpart on a node.
                                                # Should never be serialized.

    function Distributed(id, dims, pids;
                         indexes=Array(NTuple{N,UnitRange{Int}}, 0),
                         cuts=Array(Vector{Int}, 0),
                         localpart=Nullable{A}())

        if DistType == DArray
            # check invariants
            if dims != map(last, last(indexes))
                throw(ArgumentError("dimension of Distributed (dim) and indexes do not match"))
            end
        end

        release = (myid() == id[1])

        global registry
        haskey(registry, id) && return registry[id]

        d = new(id, dims, pids, indexes, cuts, release, localpart)
        if release
            push!(refs, id)
            registry[id] = d

#            println("Installing finalizer for : ", d.id, ", : ", object_id(d), ", isbits: ", isbits(d))
            finalizer(d, close)
        end
        d
    end

    Distributed() = new()
end

typealias SubDistributed{T,N,D<:Distributed} SubArray{T,N,D}
typealias SubOrDistributed{T,N} Union{Distributed{T,N}, SubDistributed{T,N}}

localtype{T,N,S}(A::Distributed{T,N,S}) = S
localtype(A::AbstractArray) = typeof(A)

# BCast is the same value broadcasted on each node
# DAny stores one element per worker
# DArray is a regular N-dimensional array distributed across participating workers

@enum DISTTYPE DArray=1 BCast=2 DAny=3

Base.show(io, disttype::DISTTYPE) = begin
    println(io, ["DArray", "BCast", "DAny"][disttype])
end

## core constructors ##

function Distributed(id, init, dims, pids, idxs, cuts, disttype=DArray)
    r=Channel(1)

    if disttype == DArray
        if isa(init, Function)
            clp_func = i -> remotecall_fetch(construct_localparts, pids[i], init, id, dims, pids, idxs, cuts)
        else
            # constructing from an array of remote refs.
            clp_func = i -> remotecall_fetch(construct_localparts, pids[i], init[i], id, dims, pids, idxs, cuts)
        end
    else
        clp_func = i -> remotecall_fetch(construct_localparts, pids[i], init, id, pids, disttype)
    end

    @sync begin
        for i = 1:length(pids)
            @async begin
                local typA
                typA=clp_func(i)
                !isready(r) && put!(r, typA)
            end
        end
    end

    typA = take!(r)

    if myid() in pids
        d = registry[id]
    else
        if disttype == DArray
            d = Distributed{eltype(typA),length(dims),typA, disttype}(id, dims, pids; indexes=idxs, cuts=cuts)
        elseif disttype == BCast
            # Keep a local copy of the broadcasted variable. Bad idea?
            construct_localparts(init, id, pids, disttype)
            d = registry[id]
        else
            d = Distributed{typA,1,typA,disttype}(id, (length(pids),), pids)
        end
    end
    d
end

function construct_localparts(init, id, dims, pids, idxs, cuts)
    A = isa(init, Function) ? init(idxs[localpartindex(pids)]) : fetch(init)
    global registry
    typA = typeof(A)
    d = Distributed{eltype(typA),length(dims),typA,DArray}(id, dims, pids;
                                                    indexes=idxs, cuts=cuts, localpart=Nullable{typA}(A))
    registry[id] = d
    typA
end

function construct_localparts(init, id, pids, disttype::DISTTYPE)
    A = isa(init, Function) ? init(localpartindex(pids)) : fetch(init)
    global registry
    typA = typeof(A)
    d = Distributed{typA,1,typA,disttype}(id, (length(pids),), pids; localpart=Nullable{typA}(A))
    registry[id] = d
    typA
end

function Distributed(init::Function, dims, procs, dist)
    np = prod(dist)
    procs = reshape(procs[1:np], ntuple(i->dist[i], length(dist)))
    idxs, cuts = chunk_idxs([dims...], dist)
    id = next_did()

    return Distributed(id, init, dims, procs, idxs, cuts)
end

function Distributed(init::Function, dims, procs)
    if isempty(procs)
        throw(ArgumentError("no processors given"))
    end
    return Distributed(init, dims, procs, defaultdist(dims, procs))
end
Distributed(init::Function, dims) = Distributed(init, dims, workers()[1:min(nworkers(), maximum(dims))])

# Create a Distributed from a collection of references
# The refs must have the same layout as the parts distributed.
# i.e.
#    size(refs) must specify the distribution of dimensions across processors
#    prod(size(refs)) must equal number of parts
# FIXME : Empty parts are currently not supported.
function Distributed(x; procs=workers(), disttype=DArray)
    if isa(x, Array) && (disttype != BCast)
        if (disttype == DArray) &&
           ((eltype(x) <: Base.AbstractRemoteRef) || all(x->isa(x, Base.AbstractRemoteRef)))
            return distributed_from_refs(x)
        else
            return distribute(x, disttype)
        end
    elseif isa(x, Function) && (disttype != DArray)
        return Distributed(next_did(), x, nothing, procs, nothing, nothing, disttype)
    else
        return distribute(x, procs, BCast)
    end
end

function distributed_from_refs(refs)
    dimdist = size(refs)
    id = next_did()

    npids = [r.where for r in refs]
    nsizes = Array(Tuple, dimdist)
    @sync for i in 1:length(refs)
        let i=i
            @async nsizes[i] = remotecall_fetch(rr_size_localpart, npids[i], refs[i])
        end
    end

    nindexes = Array(NTuple{length(dimdist),UnitRange{Int}}, dimdist...)

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

    Distributed(id, refs, ndims, reshape(npids, dimdist), nindexes, ncuts)
end


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
    return Distributed(I->localpart(s), size(A), procs_used, dist)
end

"""
    distribute(A, DA)

Distribute a local array `A` like the distributed array `DA`.

"""
function distribute(A::AbstractArray, DA::Distributed)
    size(DA) == size(A) || throw(DimensionMismatch("Distributed array has size $(size(DA)) but array has $(size(A))"))

    s = verified_destination_serializer(procs(DA), size(DA.indexes)) do pididx
        A[DA.indexes[pididx]...]
    end
    return Distributed(I->localpart(s), DA)
end

function distribute(x, procs::Array=workers(), disttype::DISTTYPE=DArray)
    if disttype == BCast
        return Distributed(next_did(), I->x, nothing, procs, nothing, nothing, BCast)
    elseif isa(x, AbstractArray) && (disttype == DAny)
        @assert length(x) == length(procs)
        s = DestinationSerializer(p->x[p], procs)
        return Distributed(next_did(), I->localpart(s), nothing, procs, nothing, nothing, DAny)

    elseif isa(x, AbstractArray)
            return distribute(x; procs=procs)
    else
        error("$(typeof(x)) cannot be distributed as $disttype")
    end
end

macro Distributed(ex0::Expr)
    if ex0.head !== :comprehension
        throw(ArgumentError("invalid @Distributed syntax"))
    end
    ex = ex0.args[1]
    if ex.head !== :generator
        throw(ArgumentError("invalid @Distributed syntax"))
    end
    ex.args[1] = esc(ex.args[1])
    ndim = length(ex.args) - 1
    ranges = map(r->esc(r.args[2]), ex.args[2:end])
    for d = 1:ndim
        var = ex.args[d+1].args[1]
        ex.args[d+1] = :( $(esc(var)) = ($(ranges[d]))[I[$d]] )
    end
    return :( Distributed((I::Tuple{Vararg{UnitRange{Int}}})->($ex0),
                tuple($(map(r->:(length($r)), ranges)...))) )
end

# new Distributed similar to an existing one
function Distributed{T,N,A,DT}(init::Function, d::Distributed{T,N,A,DT}; disttype=DT)
    if disttype == DArray
        return Distributed(next_did(), init, size(d), procs(d), d.indexes, d.cuts)
    else
        return Distributed(next_did(), init, nothing, procs(d), nothing, nothing, disttype)
    end
end

function release_localpart(id)
    global registry
    delete!(registry, id)
    nothing
end
release_localpart(d::Distributed) = release_localpart(d.id)

function close_by_id(id, pids)
#   @schedule println("Finalizer for : ", id)
    global refs
    @sync begin
        for p in pids
            @async remotecall_fetch(release_localpart, p, id)
        end
        if !(myid() in pids)
            release_localpart(id)
        end
    end
    delete!(refs, id)
    nothing
end

function close(d::Distributed)
#    @schedule println("close : ", d.id, ", object_id : ", object_id(d), ", myid : ", myid() )
    if (myid() == d.id[1]) && d.release
        @schedule close_by_id(d.id, d.pids)
        d.release = false
    end
    nothing
end

function darray_closeall()
    global registry
    global refs
    crefs = copy(refs)
    for id in crefs
        if id[1] ==  myid() # sanity check
            haskey(registry, id) && close(registry[id])
            yield()
        end
    end
end

rr_size_localpart(ref) = size(fetch(ref))

Base.similar(d::Distributed, T::Type, dims::Dims) = Distributed(I->Array(T, map(length,I)), dims, procs(d))
Base.similar(d::Distributed, T::Type) = similar(d, T, size(d))
Base.similar{T}(d::Distributed{T}, dims::Dims) = similar(d, T, dims)
Base.similar{T}(d::Distributed{T}) = similar(d, T, size(d))

Base.size(d::Distributed) = d.dims


"""
    procs(d::Distributed)

Get the vector of processes storing pieces of Distributed `d`.
"""
Base.procs(d::Distributed) = d.pids

chunktype{T,N,A}(d::Distributed{T,N,A}) = A

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
        return round(Int, linspace(1, sz+1, nc+1))
    else
        return [[1:(sz+1);], zeros(Int, nc-sz);]
    end
end

# compute indexes array for dividing dims into chunks
function chunk_idxs(dims, chunks)
    cuts = map(defaultdist, dims, chunks)
    n = length(dims)
    idxs = Array(NTuple{n,UnitRange{Int}},chunks...)
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
localpartindex(d::Distributed) = localpartindex(procs(d))

"""
    localpart(d::Distributed)

Get the local piece of a distributed array.
Returns an empty array if no local part exists on the calling process.
"""
function localpart{T,N,A,DT}(d::Distributed{T,N,A,DT})
    if isnull(d.v)
        if DT == DArray
            return convert(A, Array(T, ntuple(zero, N)))::A
        else
            error("No localpart available")
        end
    else
        return get(d.v)::A
    end
end

localpart(d::Distributed, localidx...) = localpart(d)[localidx...]

# fetch localpart of d at pids[i]
fetch{T,N,A}(d::Distributed{T,N,A}, i) = remotecall_fetch(localpart, d.pids[i], d)

"""
    localpart(A)

The identity when input is not distributed
"""
localpart(A) = A

"""
    localindexes(d)

A tuple describing the indexes owned by the local process.
Returns a tuple with empty ranges if no local part exists on the calling process.
"""
function localindexes(d::Distributed)
    lpidx = localpartindex(d)
    if lpidx == 0
        return ntuple(i -> 1:0, ndims(d))
    end
    return d.indexes[lpidx]
end

# find which piece holds index (I...)
locate(d::Distributed, I::Int...) =
    ntuple(i -> searchsortedlast(d.cuts[i], I[i]), ndims(d))

chunk{T,N,A}(d::Distributed{T,N,A}, i...) = remotecall_fetch(localpart, d.pids[i...], d)::A

## convenience constructors ##

"""
     dzeros(dims, ...)

Construct a distributed array of zeros.
Trailing arguments are the same as those accepted by `Distributed`.
"""
dzeros(dims::Dims, args...) = Distributed(I->zeros(map(length,I)), dims, args...)
dzeros{T}(::Type{T}, dims::Dims, args...) = Distributed(I->zeros(T,map(length,I)), dims, args...)
dzeros{T}(::Type{T}, d1::Integer, drest::Integer...) = dzeros(T, convert(Dims, tuple(d1, drest...)))
dzeros(d1::Integer, drest::Integer...) = dzeros(Float64, convert(Dims, tuple(d1, drest...)))
dzeros(d::Dims) = dzeros(Float64, d)


"""
    dones(dims, ...)

Construct a distributed array of ones.
Trailing arguments are the same as those accepted by `Distributed`.
"""
dones(dims::Dims, args...) = Distributed(I->ones(map(length,I)), dims, args...)
dones{T}(::Type{T}, dims::Dims, args...) = Distributed(I->ones(T,map(length,I)), dims, args...)
dones{T}(::Type{T}, d1::Integer, drest::Integer...) = dones(T, convert(Dims, tuple(d1, drest...)))
dones(d1::Integer, drest::Integer...) = dones(Float64, convert(Dims, tuple(d1, drest...)))
dones(d::Dims) = dones(Float64, d)

"""
     dfill(x, dims, ...)

Construct a distributed array filled with value `x`.
Trailing arguments are the same as those accepted by `Distributed`.
"""
dfill(v, dims::Dims, args...) = Distributed(I->fill(v, map(length,I)), dims, args...)
dfill(v, d1::Integer, drest::Integer...) = dfill(v, convert(Dims, tuple(d1, drest...)))

"""
     drand(dims, ...)

Construct a distributed uniform random array.
Trailing arguments are the same as those accepted by `Distributed`.
"""
drand{T}(::Type{T}, dims::Dims, args...) = Distributed(I->rand(T,map(length,I)), dims, args...)
drand{T}(::Type{T}, d1::Integer, drest::Integer...) = drand(T, convert(Dims, tuple(d1, drest...)))
drand(d1::Integer, drest::Integer...) = drand(Float64, convert(Dims, tuple(d1, drest...)))
drand(d::Dims, args...)  = drand(Float64, d, args...)

"""
     drandn(dims, ...)

Construct a distributed normal random array.
Trailing arguments are the same as those accepted by `Distributed`.
"""
drandn(dims::Dims, args...) = Distributed(I->randn(map(length,I)), dims, args...)
drandn(d1::Integer, drest::Integer...) = drandn(convert(Dims, tuple(d1, drest...)))

## conversions ##

Base.convert{T,N,S<:AbstractArray}(::Type{Distributed{T,N,S}}, A::S) = distribute(convert(AbstractArray{T,N}, A))

Base.convert{S,T,N}(::Type{Array{S,N}}, d::Distributed{T,N}) = begin
    a = Array(S, size(d))
    @sync begin
        for i = 1:length(d.pids)
            @async a[d.indexes[i]...] = chunk(d, i)
        end
    end
    return a
end

Base.convert{S,T,N}(::Type{Array{S,N}}, s::SubDistributed{T,N}) = begin
    I = s.indexes
    d = s.parent
    if isa(I,Tuple{Vararg{UnitRange{Int}}}) && S<:T && T<:S
        l = locate(d, map(first, I)...)
        if isequal(d.indexes[l...], I)
            # SubDistributed corresponds to a chunk
            return chunk(d, l...)
        end
    end
    a = Array(S, size(s))
    a[[1:size(a,i) for i=1:N]...] = s
    return a
end

function Base.convert{T,N}(::Type{Distributed}, SD::SubArray{T,N})
    D = SD.parent
    Distributed(size(SD), procs(D)) do I
        TR = typeof(SD.indexes[1])
        lindices = Array(TR, 0)
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

Base.reshape{T,S<:Array}(A::Distributed{T,1,S}, d::Dims) = begin
    if prod(d) != length(A)
        throw(DimensionMismatch("dimensions must be consistent with array size"))
    end
    return Distributed(d) do I
        sz = map(length,I)
        d1offs = first(I[1])
        nd = length(I)

        B = Array(T,sz)
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

getlocalindex(d::Distributed, idx...) = localpart(d)[idx...]
function getindex_tuple{T}(d::Distributed{T}, I::Tuple{Vararg{Int}})
    chidx = locate(d, I...)
    idxs = d.indexes[chidx...]
    localidx = ntuple(i -> (I[i] - first(idxs[i]) + 1), ndims(d))
    pid = d.pids[chidx...]
    return remotecall_fetch(getlocalindex, pid, d, localidx...)::T
end

Base.getindex(d::Distributed, i::Int) = getindex_tuple(d, ind2sub(size(d), i))
Base.getindex(d::Distributed, i::Int...) = getindex_tuple(d, i)

function Base.getindex{T,N,A,DT}(d::Distributed{T,N,A,DT})
    if DT == DArray
        return d[1]::A
    elseif (DT == BCast) || (DT == DAny)
        return localpart(d)::A
    end
end
Base.getindex(d::Distributed, I::Union{Int,UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...) = view(d, I...)

# get the localpart from a different worker
type DPid
    pid::Int
end

Base.getindex{T,N,A}(d::Distributed{T,N,A}, pid::DPid) = remotecall_fetch(localpart, pid.pid, d)::A

# shorter syntax for localpart
function Base.getindex{T,N,A}(d::Distributed{T,N,A}, s::Symbol)
    s != :L && error("Use d[:L] to access the localpart of a distributed type")
    localpart(d)::A
end


Base.copy!(dest::SubOrDistributed, src::SubOrDistributed) = begin
    asyncmap(procs(dest)) do p
        remotecall_fetch(p) do
            localpart(dest)[:] = src[localindexes(dest)...]
        end
    end
    return dest
end

# local copies are obtained by convert(Array, ) or assigning from
# a SubDistributed to a local Array.

function Base.setindex!(a::Array, d::Distributed,
        I::Union{UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...)
    n = length(I)
    @sync for i = 1:length(d.pids)
        K = d.indexes[i]
        @async a[[I[j][K[j]] for j=1:n]...] = chunk(d, i)
    end
    return a
end

# We also want to optimize setindex! with a SubDistributed source, but this is hard
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
typealias ReshapedMergedIndices{T,N,M<:MergedIndices} Base.ReshapedArray{T,N,M}
typealias SubMergedIndices{T,N,M<:Union{MergedIndices, ReshapedMergedIndices}} SubArray{T,N,M}
typealias MergedIndicesOrSub Union{MergedIndices, ReshapedMergedIndices, SubMergedIndices}
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
# 1. Find the indices of each portion of the Distributed.
# 2. Find the valid subset of indices for the SubArray into that portion.
# 3. Find the portion of the `I` indices that should be used when you access the
#    `K` indices in the subarray.  This guy is nasty.  It’s totally backwards
#    from all other arrays, wherein we simply iterate over the source array’s
#    elements.  You need to *both* know which elements in `J` were skipped
#    (`indexin_mask`) and which dimensions should match up (`restrict_indices`)
# 4. If `K` doesn’t correspond to an entire chunk, reinterpret `K` in terms of
#    the local portion of the source array
function Base.setindex!(a::Array, s::SubDistributed,
        I::Union{UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...)
    Base.setindex_shape_check(s, Base.index_lengths(a, I...)...)
    n = length(I)
    d = s.parent
    J = Base.decolon(d, s.indexes...)
    @sync for i = 1:length(d.pids)
        K_c = d.indexes[i]
        K = map(intersect, J, K_c)
        if !any(isempty, K)
            K_mask = map(indexin_mask, J, K_c)
            idxs = restrict_indices(Base.decolon(a, I...), K_mask)
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

Base.fill!(A::Distributed, x) = begin
    @sync for p in procs(A)
        @async remotecall_fetch((A,x)->(fill!(localpart(A), x); nothing), p, A, x)
    end
    return A
end

Base.show{T,N,A,DistType}(io::IO, m::MIME"text/plain", d::Distributed{T,N,A,DistType}) = begin
    if DistType == DArray
        invoke(show, (IO, MIME"text/plain", AbstractArray), io, m, d)
    else
        print(io, "DistributedArrays.Distributed{$T,$N,$A,$DistType}")
    end
end

# MPI type communication enablers
bcast(x; procs=workers()) = Distributed(x; procs=procs, disttype=BCast)
sendto(x, pid) = distribute(x, [pid], BCast)
recvfrom(d::Distributed, pid) = d[DPid(pid)]

#reduce is the regular reduce
all_reduce(op, d::Distributed) = distribute(reduce(op, d), procs(d), BCast)

# scatter
function scatter(x, procs=workers())
    if length(x) > length(procs)
        error(string("scatter only for works for input items smallet than the number of workers.",
                     "Use distribute(::DArray) for longer items."))
    end
    distribute(x, procs[1:length(x)], DAny)
end

function gather_data{T,N,A,DT}(d::Distributed{T,N,A,DT})
    if DT == DArray
        gv = convert(Array, d)
    else
        gv = Array(A, length(procs(d)))
        @sync for (i,p) in enumerate(procs(d))
            @async gv[i] = d[DPid(p)]
        end
        gv
    end
    return gv
end

gather(d) = Distributed(gather_data(d); disttype=BCast, procs=[myid()])

all_gather(d) = Distributed(I->gather_data(d); disttype=DAny)  # should this be the same disttype as input?

# all_to_all will work on DArray or with DAny. Both require that the localpart is a vector
# whose length is a multiple of nworkers()
function all_to_all{T,N,A,DT}(d::Distributed{T,N,A,DT})
    ((DT != DAny) || (N > 1)) && error("all_to_all in only supported for DAny types where the localpart is a vector")

    # create a DAny to store the results
    d2 = Distributed(I->Array(eltype(A), length(d[:L])); disttype=DAny, procs=procs(d))

    # run local_all_to_all part on all pids
    @sync for p in procs(d)
        @async remotecall_wait(local_all_to_all, p, d, d2)
    end
    d2
end

function local_all_to_all{T,N,A,DT}(d::Distributed{T,N,A,DT}, d2)
    lp = localpart(d)
    lp2 = localpart(d2)

    @assert isa(lp, AbstractArray) && (rem(length(lp), length(d.pids)) == 0)

    csz = div(length(lp), length(d.pids))

    # Higher pids exchange with lower pids
    np = length(procs(d))
    j = localpartindex(d)
    @sync for (i,p) in enumerate(procs(d)[1:j-1])
        send_ith_data = lp[1+(i-1)*csz : i*csz]

        @async begin
            lp2[1+(i-1)*csz : i*csz] = remotecall_fetch((d,d2,data,i,j) -> begin
                                                localpart(d2)[1+(j-1)*csz : j*csz] = data
                                                localpart(d)[1+(j-1)*csz : j*csz]
                                            end, p, d, d2, send_ith_data, i, j)
        end
    end
    # copy diag values
    lp2[1+(j-1)*csz : j*csz] = lp[1+(j-1)*csz : j*csz]
    nothing
end


export DArray, BCast, DAny, DPid, bcast, sendto, recvfrom, all_reduce, scatter, gather, all_to_all, all_gather

