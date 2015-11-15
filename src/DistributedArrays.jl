VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

# TODO
# - avoid temporary darrays being created by sum, mean, std, etc when called along specific dimensions


module DistributedArrays

using Compat

importall Base
import Base.Callable
import Base.BLAS: axpy!

export (.+), (.-), (.*), (./), (.%), (.<<), (.>>), div, mod, rem, (&), (|), ($)
export DArray, SubDArray, SubOrDArray, @DArray
export dzeros, dones, dfill, drand, drandn, distribute, localpart, localindexes, ppeval, samedist
export close, darray_closeall

const registry=Dict{Tuple, Any}()
const refs=Set()  # Collection of darray identities created on this node


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
    identity::Tuple
    dims::NTuple{N,Int}
    pids::Array{Int,N}                          # pids[i]==p ⇒ processor p has piece i
    indexes::Array{NTuple{N,UnitRange{Int}},N}  # indexes held by piece i
    cuts::Vector{Vector{Int}}                   # cuts[d][i] = first index of chunk i in dimension d

    release::Bool

    function DArray(identity, dims, pids, indexes, cuts)
        # check invariants
        if dims != map(last, last(indexes))
            throw(ArgumentError("dimension of DArray (dim) and indexes do not match"))
        end
        release = (myid() == identity[1])

        global registry
        haskey(registry, (identity, :DARRAY)) && return registry[(identity, :DARRAY)]

        d = new(identity, dims, pids, indexes, cuts, release)
        if release
            push!(refs, identity)
            registry[(identity, :DARRAY)] = d

#            println("Installing finalizer for : ", d.identity, ", : ", object_id(d), ", isbits: ", isbits(d))
            finalizer(d, close)
        end
        d
    end

    DArray() = new()
end

typealias SubDArray{T,N,D<:DArray} SubArray{T,N,D}
typealias SubOrDArray{T,N} Union{DArray{T,N}, SubDArray{T,N}}

## core constructors ##

function DArray(init, dims, procs, dist)
    np = prod(dist)
    procs = reshape(procs[1:np], ntuple(i->dist[i], length(dist)))
    idxs, cuts = chunk_idxs([dims...], dist)
    identity = next_did()

    return construct_darray(identity, init, dims, procs, idxs, cuts)
end

function DArray(init, dims, procs)
    if isempty(procs)
        throw(ArgumentError("no processors given"))
    end
    return DArray(init, dims, procs, defaultdist(dims, procs))
end
DArray(init, dims) = DArray(init, dims, workers()[1:min(nworkers(), maximum(dims))])

# Create a DArray from a collection of references
function DArray(refs::Array{RemoteRef})
    dimdist = size(refs)
    identity = next_did()

    npids = [r.where for r in refs]
    nsizes = Array(Tuple, dimdist)
    @sync for i in 1:length(refs)
        let i=i
            @async nsizes[i] = remotecall_fetch(rr_localpart, npids[i], refs[i], identity)
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

    construct_darray(identity, refs, ndims, reshape(npids, dimdist), nindexes, ncuts)
end

macro DArray(ex::Expr)
    if ex.head !== :comprehension
        throw(ArgumentError("invalid @DArray syntax"))
    end
    ex.args[1] = esc(ex.args[1])
    ndim = length(ex.args) - 1
    ranges = map(r->esc(r.args[2]), ex.args[2:end])
    for d = 1:ndim
        var = ex.args[d+1].args[1]
        ex.args[d+1] = :( $(esc(var)) = ($(ranges[d]))[I[$d]] )
    end
    return :( DArray((I::Tuple{Vararg{UnitRange{Int}}})->($ex),
                tuple($(map(r->:(length($r)), ranges)...))) )
end

# new DArray similar to an existing one
DArray(init, d::DArray) = DArray(init, size(d), procs(d), [size(d.pids)...])

function construct_darray(identity, init, dims, pids, idxs, cuts)
    r=Channel(1)
    @sync begin
        for i = 1:length(pids)
            @async begin
                local typA
                if isa(init, Function)
                    typA=remotecall_fetch(construct_localparts, pids[i], init, identity, dims, pids, idxs, cuts)
                else
                    # constructing from an array of remote refs.
                    typA=remotecall_fetch(construct_localparts, pids[i], init[i], identity, dims, pids, idxs, cuts)
                end
                !isready(r) && put!(r, typA)
            end
        end
    end

    typA = take!(r)
    if myid() in pids
        d = registry[(identity, :DARRAY)]
    else
        d = DArray{eltype(typA),length(dims),typA}(identity, dims, pids, idxs, cuts)
    end
    d
end

function construct_localparts(init, identity, dims, pids, idxs, cuts)
    A = isa(init, Function) ? init(idxs[localpartindex(pids)]) : fetch(init)
    global registry
    registry[(identity, :LOCALPART)] = A
    typA = typeof(A)
    d = DArray{eltype(typA),length(dims),typA}(identity, dims, pids, idxs, cuts)
    registry[(identity, :DARRAY)] = d
    typA
end

let DID::Int = 1
    global next_did
    next_did() = (id = DID; DID += 1; (myid(), id))
end

function release_localpart(identity)
    global registry
    delete!(registry, (identity, :DARRAY))
    delete!(registry, (identity, :LOCALPART))
    nothing
end
release_localpart(d::DArray) = release_localpart(d.identity)

function close_by_identity(identity, pids)
#   @schedule println("Finalizer for : ", identity)
    global refs
    @sync begin
        for p in pids
            @async remotecall_fetch(release_localpart, p, identity)
        end
        if !(myid() in pids)
            release_localpart(identity)
        end
    end
    delete!(refs, identity)
    nothing
end

function close(d::DArray)
#    @schedule println("close : ", d.identity, ", object_id : ", object_id(d), ", myid : ", myid() )
    if (myid() == d.identity[1]) && d.release
        @schedule close_by_identity(d.identity, d.pids)
        d.release = false
    end
    nothing
end

function darray_closeall()
    global registry
    global refs
    crefs = copy(refs)
    for identity in crefs
        if identity[1] ==  myid() # sanity check
            haskey(registry, (identity, :DARRAY)) && close(registry[(identity, :DARRAY)])
            yield()
        end
    end
end

function rr_localpart(r::RemoteRef, identity)
    global registry
    lp = fetch(r)
    registry[(identity, :LOCALPART)] = lp
    return size(lp)
end

function Base.serialize(S::SerializationState, d::DArray)
    # Only send the ident for participating workers - we expect the DArray to exist in the
    # remote registry
    destpid = Base.worker_id_from_socket(S.io)
    Serializer.serialize_type(S, typeof(d))
    if (destpid in d.pids) || (destpid == d.identity[1])
        serialize(S, (true, d.identity))    # (identity_only, identity)
    else
        serialize(S, (false, d.identity))
        for n in [:dims, :pids, :indexes, :cuts]
            serialize(S, getfield(d, n))
        end
    end
end

function Base.deserialize{T<:DArray}(S::SerializationState, t::Type{T})
    what = deserialize(S)
    identity_only = what[1]
    identity = what[2]

    if identity_only
        global registry
        if haskey(registry, (identity, :DARRAY))
            return registry[(identity, :DARRAY)]
        else
            # access to fields will throw an error, at least the deserialization process will not
            # result in worker death
            d = T()
            d.identity = identity
            return d
        end
    else
        # We are not a participating worker, deser fields and instantiate locally.
        dims = deserialize(S)
        pids = deserialize(S)
        indexes = deserialize(S)
        cuts = deserialize(S)
        return T(identity, dims, pids, indexes, cuts)
    end
end


Base.similar(d::DArray, T, dims::Dims) = DArray(I->Array(T, map(length,I)), dims, procs(d))
Base.similar(d::DArray, T) = similar(d, T, size(d))
Base.similar{T}(d::DArray{T}, dims::Dims) = similar(d, T, dims)
Base.similar{T}(d::DArray{T}) = similar(d, T, size(d))

Base.size(d::DArray) = d.dims


"""
    procs(d::DArray)

Get the vector of processes storing pieces of DArray `d`.
"""
Base.procs(d::DArray) = d.pids

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
localpartindex(d::DArray) = localpartindex(procs(d))

"""
    localpart(d::DArray)

Get the local piece of a distributed array.
Returns an empty array if no local part exists on the calling process.
"""
function localpart{T,N,A}(d::DArray{T,N,A})
    lpidx = localpartindex(d)
    if lpidx == 0
        return convert(A, Array(T, ntuple(zero, N)))::A
    end

    global registry
    return registry[(d.identity, :LOCALPART)]::A
end

localpart(d::DArray, localidx...) = localpart(d)[localidx...]

# fetch localpart of d at pids[i]
fetch{T,N,A}(d::DArray{T,N,A}, i) = remotecall_fetch(localpart, d.pids[i], d)

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
drand{T}(::Type{T}, dims::Dims, args...) = DArray(I->rand(T,map(length,I)), dims, args...)
drand{T}(::Type{T}, d1::Integer, drest::Integer...) = drand(T, convert(Dims, tuple(d1, drest...)))
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

    owner = myid()
    rr = RemoteRef()
    put!(rr, A)
    d = DArray(size(A), procs, dist) do I
        remotecall_fetch(() -> fetch(rr)[I...], owner)
    end
    return d
end

Base.convert{T,N,S<:AbstractArray}(::Type{DArray{T,N,S}}, A::S) = distribute(convert(AbstractArray{T,N}, A))

Base.convert{S,T,N}(::Type{Array{S,N}}, d::DArray{T,N}) = begin
    a = Array(S, size(d))
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
    a = Array(S, size(s))
    a[[1:size(a,i) for i=1:N]...] = s
    return a
end

function Base.convert{T,N}(::Type{DArray}, SD::SubArray{T,N})
    D = SD.parent
    DArray(SD.dims, procs(D)) do I
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

Base.reshape{T,S<:Array}(A::DArray{T,1,S}, d::Dims) = begin
    if prod(d) != length(A)
        throw(DimensionMismatch("dimensions must be consistent with array size"))
    end
    return DArray(d) do I
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
Base.getindex(d::DArray, I::Union{Int,UnitRange{Int},Colon,Vector{Int},StepRange{Int,Int}}...) = sub(d, I...)

Base.copy!(dest::SubOrDArray, src::SubOrDArray) = begin
    if !(dest.dims == src.dims &&
         procs(dest) == procs(src) &&
         dest.indexes == src.indexes &&
         dest.cuts == src.cuts)
        throw(DimensionMismatch("destination array doesn't fit to source array"))
    end
    @sync for p in procs(dest)
        @spawnat p copy!(localpart(dest), localpart(src))
    end
    return dest
end

# local copies are obtained by convert(Array, ) or assigning from
# a SubDArray to a local Array.

Base.setindex!(a::Array, d::DArray, I::UnitRange{Int}...) = begin
    n = length(I)
    @sync for i = 1:length(d.pids)
        K = d.indexes[i]
        @async a[[I[j][K[j]] for j=1:n]...] = chunk(d, i)
    end
    return a
end

Base.setindex!(a::Array, s::SubDArray, I::UnitRange{Int}...) = begin
    n = length(I)
    d = s.parent
    J = s.indexes
    if length(J) < n
        a[I...] = convert(Array,s)
        return a
    end
    offs = [isa(J[i],Int) ? J[i]-1 : first(J[i])-1 for i=1:n]
    @sync for i = 1:length(d.pids)
        K_c = Any[d.indexes[i]...]
        K = [ intersect(J[j],K_c[j]) for j=1:n ]
        if !any(isempty, K)
            idxs = [ I[j][K[j]-offs[j]] for j=1:n ]
            if isequal(K, K_c)
                # whole chunk
                @async a[idxs...] = chunk(d, i)
            else
                # partial chunk
                @async a[idxs...] =
                    remotecall_fetch(d.pids[i]) do
                        sub(localpart(d), [K[j]-first(K_c[j])+1 for j=1:n]...)
                    end
            end
        end
    end
    return a
end

# to disambiguate
Base.setindex!(a::Array{Any}, d::SubOrDArray, i::Int) = Base.arrayset(a, d, i)

Base.fill!(A::DArray, x) = begin
    @sync for p in procs(A)
        @spawnat p fill!(localpart(A), x)
    end
    return A
end

## higher-order functions ##

Base.map(f, d::DArray) = DArray(I->map(f, localpart(d)), d)

function Base.reduce(f, d::DArray)
    results=[]
    @sync begin
        for p in procs(d)
            @async push!(results, remotecall_fetch((f,d)->reduce(f, localpart(d)), p, f, d))
        end
    end
    reduce(f, results)
end

function _mapreduce(f, opt, d::DArray)
# TODO Change to an @async remotecall_fetch - will reduce one extra network hop -
# once bug in master is fixed.
    results=[]
    @sync begin
        for p in procs(d)
            @async push!(results, remotecall_fetch((f,opt,d)->mapreduce(f, opt, localpart(d)), p, f, opt, d))
        end
    end
    reduce(opt, results)
end
Base.mapreduce(f, opt::Union{Base.OrFun,Base.AndFun}, d::DArray) = _mapreduce(f, opt, d)
Base.mapreduce(f, opt::Function, d::DArray) = _mapreduce(f, Base.specialized_binary(opt), d)
Base.mapreduce(f, opt, d::DArray) = _mapreduce(f, opt, d)

 Base.map!(f, d::DArray) = begin
    @sync for p in procs(d)
        @spawnat p map!(f, localpart(d))
    end
    return d
end

# mapreducedim
Base.reducedim_initarray{R}(A::DArray, region, v0, ::Type{R}) = begin
    procsgrid = reshape(procs(A), size(A.indexes))
    gridsize = Base.reduced_dims(size(A.indexes), region)
    procsgrid = procsgrid[UnitRange{Int}[1:n for n = gridsize]...]
    return dfill(convert(R, v0), Base.reduced_dims(A, region), procsgrid, gridsize)
end
Base.reducedim_initarray{T}(A::DArray, region, v0::T) = Base.reducedim_initarray(A, region, v0, T)

Base.reducedim_initarray0{R}(A::DArray, region, v0, ::Type{R}) = begin
    procsgrid = reshape(procs(A), size(A.indexes))
    gridsize = Base.reduced_dims0(size(A.indexes), region)
    procsgrid = procsgrid[UnitRange{Int}[1:n for n = gridsize]...]
    return dfill(convert(R, v0), Base.reduced_dims0(A, region), procsgrid, gridsize)
end
Base.reducedim_initarray0{T}(A::DArray, region, v0::T) = Base.reducedim_initarray0(A, region, v0, T)

mapreducedim_within(f, op, A::DArray, region) = begin
    arraysize = [size(A)...]
    gridsize = [size(A.indexes)...]
    arraysize[[region...]] = gridsize[[region...]]
    return DArray(tuple(arraysize...), procs(A), tuple(gridsize...)) do I
        mapreducedim(f, op, localpart(A), region)
    end
end

function mapreducedim_between!(f, op, R::DArray, A::DArray, region)
    @sync for p in procs(R)
        @async remotecall_wait(p, f, op, R, A, region) do f, op, R, A, region
            localind = [r for r = localindexes(A)]
            localind[[region...]] = [1:n for n = size(A)[[region...]]]
            B = convert(Array, A[localind...])
            Base.mapreducedim!(f, op, localpart(R), B)
        end
    end
    return R
end

Base.mapreducedim!(f, op, R::DArray, A::DArray) = begin
    lsize = Base.check_reducedims(R,A)
    if isempty(A)
        return copy(R)
    end
    region = tuple(collect(1:ndims(A))[[size(R)...] .!= [size(A)...]]...)
    if isempty(region)
        return copy!(R, A)
    end
    B = mapreducedim_within(f, op, A, region)
    return mapreducedim_between!(identity, op, R, B, region)
end

Base.mapreducedim(f, op, R::DArray, A::DArray) = begin
    Base.mapreducedim!(f, op, Base.reducedim_initarray(A, region, v0), A)
end

function nnz(A::DArray)
    B = Array(Any, size(A.pids))
    @sync begin
        for i in eachindex(A.pids)
            @async B[i...] = remotecall_fetch(x -> nnz(localpart(x)), A.pids[i...], A)
        end
    end
    return reduce(+, B)
end

# LinAlg
Base.scale!(A::DArray, x::Number) = begin
    @sync for p in procs(A)
        @spawnat p scale!(localpart(A), x)
    end
    return A
end

# reduce like
for (fn, fr) in ((:sum, :AddFun),
                 (:prod, :MulFun),
                 (:maximum, :MaxFun),
                 (:minimum, :MinFun),
                 (:any, :OrFun),
                 (:all, :AndFun))
    @eval (Base.$fn)(d::DArray) = reduce((Base.$fr)(), d)
end

# mapreduce like
for (fn, fr1, fr2) in ((:maxabs, :AbsFun, :MaxFun),
                       (:minabs, :AbsFun, :MinFun),
                       (:sumabs, :AbsFun, :AddFun),
                       (:sumabs2, :Abs2Fun, :AddFun))
    @eval (Base.$fn)(d::DArray) = mapreduce((Base.$fr1)(), (Base.$fr2)(), d)
end

# semi mapreduce
for (fn, fr) in ((:any, :OrFun),
                 (:all, :AndFun),
                 (:count, :AddFun))
    @eval begin
        (Base.$fn)(f::Base.IdFun, d::DArray) = mapreduce(f, (Base.$fr)(), d)
        (Base.$fn)(f::Base.Predicate, d::DArray) = mapreduce(f, (Base.$fr)(), d)
        (Base.$fn)(f::Base.Func{1}, d::DArray) = mapreduce(f, (Base.$fr)(), d)
        (Base.$fn)(f::Callable, d::DArray) = mapreduce(f, (Base.$fr)(), d)
    end
end

# scalar ops
(+)(A::DArray{Bool}, x::Bool) = A .+ x
(+)(x::Bool, A::DArray{Bool}) = x .+ A
(-)(A::DArray{Bool}, x::Bool) = A .- x
(-)(x::Bool, A::DArray{Bool}) = x .- A
(+)(A::DArray, x::Number) = A .+ x
(+)(x::Number, A::DArray) = x .+ A
(-)(A::DArray, x::Number) = A .- x
(-)(x::Number, A::DArray) = x .- A

mappart(f::Callable, d::DArray) = DArray(i->f(localpart(d)), d)
mappart(f::Callable, d1::DArray, d2::DArray) = DArray(d1.dims, procs(d1)) do I
    f(localpart(d1), localpart(d2))
end

# Here we assume all the DArrays have
# the same size and distribution
mappart(f::Callable, As::DArray...) = DArray(I->f(map(localpart, As)...), As[1])

for f in (:.+, :.-, :.*, :./, :.%, :.<<, :.>>, :div, :mod, :rem, :&, :|, :$)
    @eval begin
        ($f){T}(A::DArray{T}, B::Number) = mappart(r->($f)(r, B), A)
        ($f){T}(A::Number, B::DArray{T}) = mappart(r->($f)(r, A), B)
    end
end

function samedist(A::DArray, B::DArray)
    (size(A) == size(B)) || throw(DimensionMismatch())
    if (procs(A) != procs(B)) || (A.cuts != B.cuts)
        B = DArray(x->B[x...], A)
    end
    B
end

for f in (:+, :-, :.+, :.-, :.*, :./, :.%, :.<<, :.>>, :div, :mod, :rem, :&, :|, :$)
    @eval begin
        function ($f){T}(A::DArray{T}, B::DArray{T})
            B = samedist(A, B)
            mappart($f, A, B)
        end
    end
end

function Base.ctranspose{T}(D::DArray{T,2})
    DArray(reverse(D.dims), procs(D)) do I
        lp = Array(T, map(length, I))
        rp = convert(Array, D[reverse(I)...])
        ctranspose!(lp, rp)
    end
end

function Base.transpose{T}(D::DArray{T,2})
    DArray(reverse(D.dims), procs(D)) do I
        lp = Array(T, map(length, I))
        rp = convert(Array, D[reverse(I)...])
        transpose!(lp, rp)
    end
end

for f in (:abs, :abs2, :acos, :acosd, :acosh, :acot, :acotd, :acoth,
          :acsc, :acscd, :acsch, :angle, :asec, :asecd, :asech, :asin,
          :asind, :asinh, :atan, :atand, :atanh, :big, :cbrt, :ceil, :cis,
          :complex, :cos, :cosc, :cosd, :cosh, :cospi, :cot, :cotd, :coth,
          :csc, :cscd, :csch, :dawson, :deg2rad, :digamma, :erf, :erfc,
          :erfcinv, :erfcx, :erfi, :erfinv, :exp, :exp10, :exp2, :expm1,
          :exponent, :float, :floor, :gamma, :imag, :invdigamma, :isfinite,
          :isinf, :isnan, :lfact, :lgamma, :log, :log10, :log1p, :log2, :rad2deg,
          :real, :sec, :secd, :sech, :sign, :sin, :sinc, :sind, :sinh, :sinpi,
          :sqrt, :tan, :tand, :tanh, :trigamma)
    @eval begin
        ($f)(A::DArray) = map($f, A)
    end
end

function mapslices{T,N}(f::Function, D::DArray{T,N}, dims::AbstractVector)
    #Ensure that the complete DArray is available on the specified dims on all processors
    for d in dims
        for idxs in D.indexes
            if length(idxs[d]) != size(D, d)
                throw(DimensionMismatch(string("dimension $d is distributed. ",
                    "mapslices requires dimension $d to be completely available on all processors.")))
            end
        end
    end

    refs = RemoteRef[remotecall((x,y,z)->mapslices(x,localpart(y),z), p, f, D, dims) for p in procs(D)]

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
        push!(idx, Any[1:size(A[i], d) for d in 1:dims[i]])
        if dim[i] > 0
            idx[i][dim[i]] = 1
            push!(args, slice(A[i], idx[i]...))
        else
            push!(args, A[i])
        end
    end
    R1 = f(args...)
    ridx = Any[1:size(R1, d) for d in 1:ndims(R1)]
    push!(ridx, 1)
    Rsize = map(last, ridx)
    Rsize[end] = dimlength
    R = Array(eltype(R1), Rsize...)

    for i = 1:dimlength
        for j = 1:narg
            if dim[j] > 0
                idx[j][dim[j]] = i
                args[j] = slice(A[j], idx[j]...)
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

`D` has any number of elements and the alements can have any type. If an element
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
sizes correspond to the sizes of return values of `f`. The last dimention of
the return array from `ppeval` has the same length as the dimension over which
the input arrays are sliced.

#### Examples
```jl
addprocs(JULIA_CPU_CORES)

using DistributedArrays

A = drandn((10, 10, JULIA_CPU_CORES), workers(), [1, 1, JULIA_CPU_CORES])

ppeval(eigvals, A)

ppeval(eigvals, A, randn(10,10)) # broadcasting second argument

B = drandn((10, JULIA_CPU_CORES), workers(), [1, JULIA_CPU_CORES])

ppeval(*, A, B)
```
"""
function ppeval(f, D...; dim::NTuple = map(t -> isa(t, DArray) ? ndims(t) : 0, D))
    #Ensure that the complete DArray is available on the specified dims on all processors
    for i = 1:length(D)
        if isa(D[i], DArray)
            for idxs in D[i].indexes
                for d in setdiff(1:ndims(D[i]), dim[i])
                    if length(idxs[d]) != size(D[i], d)
                        throw(DimensionMismatch(string("dimension $d is distributed. ",
                            "ppeval requires dimension $d to be completely available on all processors.")))
                    end
                end
            end
        end
    end

    refs = RemoteRef[remotecall((x, y, z) -> _ppeval(x, map(localpart, y)...; dim = z), p, f, D, dim) for p in procs(D[1])]

    # The array of RemoteRefs has to be reshaped for the DArray constructor to work correctly.
    # This requires a fetch and the DArray is also fetching so it might be better to modify
    # the DArray constructor.
    sd = [size(D[1].pids)...]
    DArray(reshape(refs, tuple([sd[1:ndims(fetch(refs[1])) - 1], sd[end];]...)))
end

typealias DVector{T,A} DArray{T,1,A}
typealias DMatrix{T,A} DArray{T,2,A}

# Level 1

function axpy!(α, x::DVector, y::DVector)
    if length(x) != length(y)
        throw(DimensionMismatch("vectors must have same length"))
    end
    @sync for p in procs(y)
        @async remotecall_wait(() -> Base.axpy!(α, localpart(x), localpart(y)), p)
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
    r = RemoteRef[]
    for i = eachindex(x.pids)
        px, py = x.pids[i], y.pids[i]
        push!(r, remotecall((x, y, i) -> dot(localpart(x), fetch(y, i)), px, x, y, i))
    end
    return mapreduce(fetch, Base.AddFun(), r)
end

function norm(x::DVector, p::Number = 2)
    r = [remotecall(() -> norm(localpart(x), p), pp) for pp in procs(x)]
    return norm([fetch(rr) for rr in r], p)
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

localtype{T,N,S}(A::DArray{T,N,S}) = S
localtype(A::AbstractArray) = typeof(A)

function A_mul_B!(α::Number, A::DMatrix, x::AbstractVector, β::Number, y::DVector)

    # error checks
    if size(A, 2) != length(x)
        throw(DimensionMismatch(""))
    end
    if y.cuts[1] != A.cuts[1]
        throw(ArgumentError("cuts of output vector must match cuts of first dimension of matrix"))
    end

    # Multiply on each tile of A
    R = Array(RemoteRef, size(A.pids)...)
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
                @async remotecall_wait(y -> scale!(localpart(y), β), p, y)
            else
                @async remotecall_wait(y -> fill!(localpart(y), 0), p, y)
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

function Ac_mul_B!(α::Number, A::DMatrix, x::AbstractVector, β::Number, y::DVector)

    # error checks
    if size(A, 1) != length(x)
        throw(DimensionMismatch(""))
    end
    if y.cuts[1] != A.cuts[2]
        throw(ArgumentError("cuts of output vector must match cuts of second dimension of matrix"))
    end

    # Multiply on each tile of A
    R = Array(RemoteRef, reverse(size(A.pids))...)
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
                @async remotecall_wait(() -> scale!(localpart(y), β), p)
            else
                @async remotecall_wait(() -> fill!(localpart(y), 0), p)
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


# Level 3
function A_mul_B!(α::Number, A::DMatrix, B::AbstractMatrix, β::Number, C::DMatrix)

    # error checks
    if size(A, 2) != size(B, 1)
        throw(DimensionMismatch(""))
    end
    if C.cuts[1] != A.cuts[1]
        throw(ArgumentError("cuts of the first dimension of the output matrix must match cuts of first dimension of the first input matrix"))
    end

    # Multiply on each tile of A
    R = Array(RemoteRef, size(procs(A))..., size(procs(C), 2))
    for j = 1:size(A.pids, 2)
        for k = 1:size(C.pids, 2)
            Bjk = B[A.cuts[2][j]:A.cuts[2][j + 1] - 1, C.cuts[2][k]:C.cuts[2][k + 1] - 1]
            for i = 1:size(A.pids, 1)
                R[i,j,k] = remotecall(procs(A)[i,j]) do
                    localpart(A)*convert(localtype(B), Bjk)
                end
            end
        end
    end

    # Scale C if necessary
    if β != one(β)
        @sync for p in C.pids
            if β != zero(β)
                @async remotecall_wait(() -> scale!(localpart(C), β), p)
            else
                @async remotecall_wait(() -> fill!(localpart(C), 0), p)
            end
        end
    end

    # Update C
    @sync for i = 1:size(R, 1)
        for k = 1:size(C.pids, 2)
            p = C.pids[i,k]
            for j = 1:size(R, 2)
                rijk = R[i,j,k]
                @async remotecall_wait(() -> add!(localpart(C), fetch(rijk), α), p)
            end
        end
    end
    return C
end

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

function Ac_mul_B!(α::Number, A::DMatrix, B::AbstractMatrix, β::Number, C::DMatrix)

    # error checks
    if size(A, 1) != size(B, 1)
        throw(DimensionMismatch(""))
    end
    if C.cuts[1] != A.cuts[2]
        throw(ArgumentError("cuts of the first dimension of the output matrix must match cuts of second dimension of the first input matrix"))
    end

    # Multiply on each tile of A
    R = Array(RemoteRef, reverse(size(procs(A)))..., size(procs(C), 2))
    for j = 1:size(A.pids, 1)
        for k = 1:size(C.pids, 2)
            Bjk = B[A.cuts[1][j]:A.cuts[1][j + 1] - 1, C.cuts[2][k]:C.cuts[2][k + 1] - 1]
            for i = 1:size(A.pids, 2)
                R[i,j,k] = remotecall(procs(A)[j,i]) do
                    localpart(A)'*convert(localtype(B), Bjk)
                end
            end
        end
    end

    # Scale C if necessary
    if β != one(β)
        @sync for p in C.pids
            if β != zero(β)
                @async remotecall_wait(() -> scale!(localpart(C), β), p)
            else
                @async remotecall_wait(() -> fill!(localpart(C), 0), p)
            end
        end
    end

    # Update C
    @sync for i = 1:size(R, 1)
        for k = 1:size(C.pids, 2)
            rsik = C.pids[i,k]
            for j = 1:size(R, 2)
                rijk = R[i,j,k]
                @async remotecall_wait(d -> add!(localpart(d), fetch(rijk), α), rsik, C)
            end
        end
    end
    return C
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

end # module
