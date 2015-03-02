module DistributedArrays

export DArray, SubOrDArray
export dzeros, dones, dfill, drand, drandn, distribute, localpart, localindexes

@doc """
### DArray(init, dims, [procs, dist])

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

```julia
dfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)
```
""" ->
type DArray{T,N,A} <: AbstractArray{T,N}
    dims::NTuple{N,Int}

    chunks::Array{RemoteRef,N}

    # pmap[i]==p ⇒ processor p has piece i
    pmap::Array{Int,N}

    # indexes held by piece i
    indexes::Array{NTuple{N,UnitRange{Int}},N}

    # cuts[d][i] = first index of chunk i in dimension d
    cuts::Vector{Vector{Int}}

    function DArray(dims, chunks, pmap, indexes, cuts)
        # check invariants
        assert(size(chunks) == size(indexes))
        assert(length(chunks) == length(pmap))
        assert(dims == map(last,last(indexes)))
        new(dims, chunks, reshape(pmap, size(chunks)), indexes, cuts)
    end
end

typealias SubDArray{T,N,D<:DArray} SubArray{T,N,D}
typealias SubOrDArray{T,N}         Union(DArray{T,N}, SubDArray{T,N})

## core constructors ##

function DArray(init, dims, procs, dist)
    # dist == size(chunks)
    np = prod(dist)
    procs = procs[1:np]
    idxs, cuts = chunk_idxs([dims...], dist)
    chunks = Array(RemoteRef, dist...)
    for i = 1:np
        chunks[i] = remotecall(procs[i], init, idxs[i])
    end
    p = max(1, localpartindex(procs))
    A = remotecall_fetch(procs[p], r->typeof(fetch(r)), chunks[p])
    DArray{eltype(A),length(dims),A}(dims, chunks, procs, idxs, cuts)
end

function DArray(init, dims, procs)
    if isempty(procs)
        throw(ArgumentError("no processors given"))
    end
    DArray(init, dims, procs, defaultdist(dims,procs))
end
DArray(init, dims) = DArray(init, dims, workers()[1:min(nworkers(),maximum(dims))])

# new DArray similar to an existing one
DArray(init, d::DArray) = DArray(init, size(d), procs(d), [size(d.chunks)...])

Base.similar(d::DArray, T, dims::Dims) = DArray(I->Array(T, map(length,I)), dims, procs(d))
Base.similar(d::DArray, T) = similar(d, T, size(d))
Base.similar{T}(d::DArray{T}, dims::Dims) = similar(d, T, dims)
Base.similar{T}(d::DArray{T}) = similar(d, T, size(d))

Base.size(d::DArray) = d.dims

@doc """
### procs(d::DArray)

Get the vector of processes storing pieces of DArray `d`.
""" ->
Base.procs(d::DArray) = d.pmap

chunktype{T,N,A}(d::DArray{T,N,A}) = A

## chunk index utilities ##

# decide how to divide each dimension
# returns size of chunks array
function defaultdist(dims, procs)
    dims = [dims...]
    chunks = ones(Int, length(dims))
    np = length(procs)
    f = sort!(collect(keys(factor(np))), rev=true)
    k = 1
    while np > 1
        # repeatedly allocate largest factor to largest dim
        if np%f[k] != 0
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
        np = div(np,fac)
    end
    chunks
end

# get array of start indexes for dividing sz into nc chunks
function defaultdist(sz::Int, nc::Int)
    if sz >= nc
        return round(Int,linspace(1, sz+1, nc+1))
    else
        return [[1:(sz+1)], zeros(Int, nc-sz)]
    end
end

# compute indexes array for dividing dims into chunks
function chunk_idxs(dims, chunks)
    cuts = map(defaultdist, dims, chunks)
    n = length(dims)
    idxs = Array(NTuple{n,UnitRange{Int}},chunks...)
    cartesianmap(tuple(chunks...)) do cidx...
        idxs[cidx...] = ntuple(n, i->(cuts[i][cidx[i]]:cuts[i][cidx[i]+1]-1))
    end
    idxs, cuts
end

function localpartindex(pmap::Array{Int})
    mi = myid()
    for i = 1:length(pmap)
        if pmap[i] == mi
            return i
        end
    end
    return 0
end
localpartindex(d::DArray) = localpartindex(d.pmap)

@doc """
### localpart(d)

Get the local piece of a distributed array.
Returns an empty array if no local part exists on the calling process.
""" ->
function localpart{T,N,A}(d::DArray{T,N,A})
    lpidx = localpartindex(d)
    if lpidx == 0
        convert(A, Array(T, ntuple(N,i->0)))::A
    else
        fetch(d.chunks[lpidx])::A
    end
end

@doc """
### localindexes(d)

A tuple describing the indexes owned by the local process.
Returns a tuple with empty ranges if no local part exists on the calling process.
""" ->
function localindexes(d::DArray)
    lpidx = localpartindex(d)
    if lpidx == 0
        ntuple(ndims(d), i->1:0)
    else
        d.indexes[lpidx]
    end
end

# find which piece holds index (I...)
function locate(d::DArray, I::Int...)
    ntuple(ndims(d), i->searchsortedlast(d.cuts[i], I[i]))
end

chunk{T,N,A}(d::DArray{T,N,A}, i...) = fetch(d.chunks[i...])::A

## convenience constructors ##

@doc """
### dzeros(dims, ...)

Construct a distributed array of zeros.
Trailing arguments are the same as those accepted by `DArray`.
""" ->
dzeros(args...) = DArray(I->zeros(map(length,I)), args...)
dzeros(d::Int...) = dzeros(d)

@doc """
### dzeros(dims, ...)

Construct a distributed array of ones.
Trailing arguments are the same as those accepted by `DArray`.
""" ->
dones(args...) = DArray(I->ones(map(length,I)), args...)
dones(d::Int...) = dones(d)

@doc """
### dfill(x, dims, ...)

Construct a distributed array filled with value `x`.
Trailing arguments are the same as those accepted by `DArray`.
""" ->
dfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)
dfill(v, d::Int...) = dfill(v, d)

@doc """
### drand(dims, ...)

Construct a distributed uniform random array.
Trailing arguments are the same as those accepted by `DArray`.
""" ->
drand(args...)  = DArray(I->rand(map(length,I)), args...)
drand(d::Int...) = drand(d)

@doc """
### drandn(dims, ...)

Construct a distributed normal random array.
Trailing arguments are the same as those accepted by `DArray`.
""" ->
drandn(args...) = DArray(I->randn(map(length,I)), args...)
drandn(d::Int...) = drandn(d)

## conversions ##

@doc """
### distribute(a)

Convert a local array to distributed.
""" ->
function distribute(a::AbstractArray)
    owner = myid()
    rr = RemoteRef()
    put!(rr, a)
    d = DArray(size(a)) do I
        remotecall_fetch(owner, ()->fetch(rr)[I...])
    end
    # Ensure that all workers have fetched their localparts.
    # Else a gc in between can recover the RemoteRef rr
    for chunk in d.chunks
        wait(chunk)
    end
    d
end

Base.convert{S,T,N}(::Type{Array{S,N}}, d::DArray{T,N}) = begin
    a = Array(S, size(d))
    @sync begin
        for i = 1:length(d.chunks)
            @async a[d.indexes[i]...] = chunk(d, i)
        end
    end
    return a
end

Base.convert{S,T,N}(::Type{Array{S,N}}, s::SubDArray{T,N}) = begin
    I = s.indexes
    d = s.parent
    if isa(I,(UnitRange{Int}...)) && S<:T && T<:S
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

Base.reshape{T,S<:Array}(A::DArray{T,1,S}, d::Dims) = begin
    if prod(d) != length(A)
        throw(DimensionMismatch("dimensions must be consistent with array size"))
    end
    DArray(d) do I
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

Base.getindex(r::RemoteRef, args...) = begin
    if r.where == myid()
        return getindex(fetch(r), args...)
    end
    return remotecall_fetch(r.where, getindex, r, args...)
end

function getindex_tuple{T}(d::DArray{T}, I::(Int...))
    chidx = locate(d, I...)
    chunk = d.chunks[chidx...]
    idxs = d.indexes[chidx...]
    localidx = ntuple(ndims(d), i->(I[i]-first(idxs[i])+1))
    chunk[localidx...]::T
end

Base.getindex(d::DArray, i::Int) = getindex_tuple(d, ind2sub(size(d), i))
Base.getindex(d::DArray, i::Int...) = getindex_tuple(d, i)

Base.getindex(d::DArray) = d[1]
Base.getindex(d::DArray, I::Union(Int,UnitRange{Int})...) = sub(d,I...)

Base.copy!(dest::SubOrDArray, src::SubOrDArray) = begin
    if !(dest.dims == src.dims &&
         dest.pmap == src.pmap &&
         dest.indexes == src.indexes &&
         dest.cuts == src.cuts)
        throw(DimensionMismatch("destination array doesn't fit to source array"))
    end
    @sync begin
        for p in dest.pmap
            @spawnat p copy!(localpart(dest), localpart(src))
        end
    end
    dest
end

# local copies are obtained by convert(Array, ) or assigning from
# a SubDArray to a local Array.

Base.setindex!(a::Array, d::DArray, I::UnitRange{Int}...) = begin
    n = length(I)
    @sync begin
        for i = 1:length(d.chunks)
            K = d.indexes[i]
            @async a[[I[j][K[j]] for j=1:n]...] = chunk(d, i)
        end
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
    @sync begin
        for i = 1:length(d.chunks)
            K_c = Any[d.indexes[i]...]
            K = [ intersect(J[j],K_c[j]) for j=1:n ]
            if !any(isempty, K)
                idxs = [ I[j][K[j]-offs[j]] for j=1:n ]
                if isequal(K, K_c)
                    # whole chunk
                    @async a[idxs...] = chunk(d, i)
                else
                    # partial chunk
                    ch = d.chunks[i]
                    @async a[idxs...] =
                        remotecall_fetch(ch.where,
                                         ()->sub(fetch(ch),
                                         [K[j]-first(K_c[j])+1 for j=1:n]...))
                end
            end
        end
    end
    return a
end

# to disambiguate
Base.setindex!(a::Array{Any}, d::SubOrDArray, i::Int) = arrayset(a, d, i)
Base.setindex!(a::Array, d::SubOrDArray, I::Union(Int,UnitRange{Int})...) =
    setindex!(a, d, [isa(i,Int) ? (i:i) : i for i in I ]...)


Base.fill!(A::DArray, x) = begin
    @sync for p in procs(A)
        @spawnat p fill!(localpart(A), x)
    end
    return A
end

## higher-order functions ##

Base.map(f, d::DArray) = DArray(I->map(f, localpart(d)), d)

Base.reduce(f, d::DArray) =
    mapreduce(fetch, f,
              Any[ @spawnat p reduce(f, localpart(d)) for p in procs(d) ])

Base.mapreduce(f, opt::Function, d::DArray) =
    mapreduce(fetch, opt,
              Any[ @spawnat p mapreduce(f, opt, localpart(d)) for p in procs(d) ])

Base.map!(f, d::DArray) = begin
    @sync begin
        for p in procs(d)
            @spawnat p map!(f, localpart(d))
        end
    end
    return d
end

# mapreducedim
function reducedim_initarray{R}(A::DArray, region, v0, ::Type{R})
    procsgrid = reshape(procs(A), size(A.indexes))
    gridsize = reduced_dims(size(A.indexes), region)
    procsgrid = procsgrid[UnitRange{Int}[1:n for n = gridsize]...]
    return dfill(convert(R, v0), reduced_dims(A, region), procsgrid, gridsize)
end
reducedim_initarray{T}(A::DArray, region, v0::T) = reducedim_initarray(A, region, v0, T)

function reducedim_initarray0{R}(A::DArray, region, v0, ::Type{R})
    procsgrid = reshape(procs(A), size(A.indexes))
    gridsize = reduced_dims0(size(A.indexes), region)
    procsgrid = procsgrid[UnitRange{Int}[1:n for n = gridsize]...]
    return dfill(convert(R, v0), reduced_dims0(A, region), procsgrid, gridsize)
end
reducedim_initarray0{T}(A::DArray, region, v0::T) = reducedim_initarray0(A, region, v0, T)

function mapreducedim_within(f, op, A::DArray, region)
    arraysize = [size(A)...]
    gridsize = [size(A.indexes)...]
    arraysize[[region...]] = gridsize[[region...]]
    DArray(tuple(arraysize...), procs(A), tuple(gridsize...)) do I
        mapreducedim(f, op, localpart(A), region)
    end
end

function mapreducedim_between!(f, op, R::DArray, A::DArray, region)
    @sync begin
        for p in procs(R)
            @spawnat p begin
                localind = [r for r = localindexes(A)]
                localind[[region...]] = [1:n for n = size(A)[[region...]]]
                B = convert(Array, A[localind...])
                mapreducedim!(f, op, localpart(R), B)
            end
        end
    end
    R
end

function mapreducedim!(f, op, R::DArray, A::DArray)
    nd = ndims(A)
    if nd != ndims(R)
        throw(ArgumentError("input and output arrays must have the same number of dimensions"))
    end
    region = tuple([1:ndims(A);][[size(R)...] .!= [size(A)...]]...)
    B = mapreducedim_within(f, op, A, region)
    return mapreducedim_between!(identity, op, R, B, region)
end

# LinAlg
Base.scale!(A::DArray, x::Number) = begin
    @sync for p in procs(A)
        @spawnat p scale!(localpart(A), x)
    end
    return A
end

end # module
