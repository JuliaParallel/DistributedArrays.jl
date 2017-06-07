## higher-order functions ##

Base.map(f, d::DArray) = DArray(I->map(f, localpart(d)), d)

Base.map!{F}(f::F, dest::DArray, src::DArray) = begin
    @sync for p in procs(dest)
        @async remotecall_fetch(() -> (map!(f, localpart(dest), src[localindexes(dest)...]); nothing), p)
    end
    return dest
end

Base.Broadcast._containertype{D<:DArray}(::Type{D}) = DArray

Base.Broadcast.promote_containertype(::Type{DArray}, ::Type{DArray}) = DArray
Base.Broadcast.promote_containertype(::Type{DArray}, ::Type{Array})  = DArray
Base.Broadcast.promote_containertype(::Type{DArray}, ct)             = DArray
Base.Broadcast.promote_containertype(::Type{Array}, ::Type{DArray})  = DArray
Base.Broadcast.promote_containertype(ct, ::Type{DArray})             = DArray

Base.Broadcast.broadcast_indices(::Type{DArray}, A)      = indices(A)
Base.Broadcast.broadcast_indices(::Type{DArray}, A::Ref) = ()

# FixMe!
## 1. Support for arbitrary indices including OneTo
## 2. This is as type unstable as it can be. Overhead might not matter too much for DArrays though.
function Base.Broadcast.broadcast_c(f, ::Type{DArray}, As...)
    T     = Base.Broadcast._broadcast_eltype(f, As...)
    shape = Base.Broadcast.broadcast_indices(As...)
    iter  = Base.CartesianRange(shape)
    D     = DArray(map(length, shape)) do I
        Base.Broadcast.broadcast_c(f, Array,
            map(a -> isa(a, Union{Number,Ref}) ? a :
                localtype(a)(a[ntuple(i -> i > ndims(a) ? 1 : (size(a, i) == 1 ? (1:1) : I[i]), length(shape))...]), As)...)
    end
    return D
end

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
Base.mapreduce(f, opt::Union{typeof(|), typeof(&)}, d::DArray) = _mapreduce(f, opt, d)
Base.mapreduce(f, opt::Function, d::DArray) = _mapreduce(f, opt, d)
Base.mapreduce(f, opt, d::DArray) = _mapreduce(f, opt, d)

# mapreducedim
Base.reducedim_initarray{R}(A::DArray, region, v0, ::Type{R}) = begin
    # Store reduction on lowest pids
    pids = A.pids[ntuple(i -> i in region ? (1:1) : (:), ndims(A))...]
    chunks = similar(pids, Future)
    @sync for i in eachindex(pids)
        @async chunks[i...] = remotecall_wait(() -> Base.reducedim_initarray(localpart(A), region, v0, R), pids[i...])
    end
    return DArray(chunks)
end
Base.reducedim_initarray{T}(A::DArray, region, v0::T) = Base.reducedim_initarray(A, region, v0, T)

Base.reducedim_initarray0{R}(A::DArray, region, v0, ::Type{R}) = begin
    # Store reduction on lowest pids
    pids = A.pids[ntuple(i -> i in region ? (1:1) : (:), ndims(A))...]
    chunks = similar(pids, Future)
    @sync for i in eachindex(pids)
        @async chunks[i...] = remotecall_wait(() -> Base.reducedim_initarray0(localpart(A), region, v0, R), pids[i...])
    end
    return DArray(chunks)
end
Base.reducedim_initarray0{T}(A::DArray, region, v0::T) = Base.reducedim_initarray0(A, region, v0, T)

# Compute mapreducedim of each localpart and store the result in a new DArray
mapreducedim_within(f, op, A::DArray, region) = begin
    arraysize = [size(A)...]
    gridsize = [size(A.indexes)...]
    arraysize[[region...]] = gridsize[[region...]]
    indx = similar(A.indexes)
    for i in CartesianRange(size(indx))
        indx[i] = ntuple(j -> j in region ? (i.I[j]:i.I[j]) : A.indexes[i][j], ndims(A))
    end
    cuts = [i in region ? collect(1:arraysize[i] + 1) : A.cuts[i] for i in 1:ndims(A)]
    return DArray(next_did(), I -> mapreducedim(f, op, localpart(A), region),
        tuple(arraysize...), procs(A), indx, cuts)
end

# Compute mapreducedim accros the processes. This should be done after mapreducedim
# has been run on each localpart with mapreducedim_within. Eventually, we might
# want to write mapreducedim_between! as a binary reduction.
function mapreducedim_between!(f, op, R::DArray, A::DArray, region)
    @sync for p in procs(R)
        @async remotecall_fetch(p, f, op, R, A, region) do f, op, R, A, region
            localind = [r for r = localindexes(A)]
            localind[[region...]] = [1:n for n = size(A)[[region...]]]
            B = convert(Array, A[localind...])
            Base.mapreducedim!(f, op, localpart(R), B)
            nothing
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
    B = Array{Any}(size(A.pids))
    @sync begin
        for i in eachindex(A.pids)
            @async B[i...] = remotecall_fetch(x -> nnz(localpart(x)), A.pids[i...], A)
        end
    end
    return reduce(+, B)
end

# reduce like
for (fn, fr) in ((:sum, :+),
                 (:prod, :*),
                 (:maximum, :max),
                 (:minimum, :min),
                 (:any, :|),
                 (:all, :&))
    @eval (Base.$fn)(d::DArray) = reduce($fr, d)
end

function Base.extrema(d::DArray)
    r = asyncmap(procs(d)) do p
        remotecall_fetch(p) do
            extrema(localpart(d))
        end
    end
    return reduce((t,s) -> (min(t[1], s[1]), max(t[2], s[2])), r)
end

# mapreduce like
for (fn, fr1, fr2) in ((:maxabs, :abs, :max),
                       (:minabs, :abs, :min),
                       (:sumabs, :abs, :+),
                       (:sumabs2, :abs2, :+))
    @eval (Base.$fn)(d::DArray) = mapreduce($fr1, $fr2, d)
end

# semi mapreduce
for (fn, fr) in ((:any, :|),
                 (:all, :&),
                 (:count, :+))
    @eval begin
        (Base.$fn)(f::typeof(identity), d::DArray) = mapreduce(f, $fr, d)
        (Base.$fn)(f::Callable, d::DArray) = mapreduce(f, $fr, d)
    end
end

# Unary vector functions
(-)(D::DArray) = map(-, D)

@static if VERSION < v"0.6.0-dev.1731"
    # scalar ops
    (+)(A::DArray{Bool}, x::Bool) = A .+ x
    (+)(x::Bool, A::DArray{Bool}) = x .+ A
    (-)(A::DArray{Bool}, x::Bool) = A .- x
    (-)(x::Bool, A::DArray{Bool}) = x .- A
    (+)(A::DArray, x::Number) = A .+ x
    (+)(x::Number, A::DArray) = x .+ A
    (-)(A::DArray, x::Number) = A .- x
    (-)(x::Number, A::DArray) = x .- A
end

map_localparts(f::Callable, d::DArray) = DArray(i->f(localpart(d)), d)
map_localparts(f::Callable, d1::DArray, d2::DArray) = DArray(d1) do I
    f(localpart(d1), localpart(d2))
end

function map_localparts(f::Callable, DA::DArray, A::Array)
    s = verified_destination_serializer(procs(DA), size(DA.indexes)) do pididx
        A[DA.indexes[pididx]...]
    end
    DArray(DA) do I
        f(localpart(DA), localpart(s))
    end
end

function map_localparts(f::Callable, A::Array, DA::DArray)
    s = verified_destination_serializer(procs(DA), size(DA.indexes)) do pididx
        A[DA.indexes[pididx]...]
    end
    DArray(DA) do I
        f(localpart(s), localpart(DA))
    end
end

function map_localparts!(f::Callable, d::DArray)
    @sync for p in procs(d)
        @async remotecall_fetch((f,d)->(f(localpart(d)); nothing), p, f, d)
    end
    return d
end

# Here we assume all the DArrays have
# the same size and distribution
map_localparts(f::Callable, As::DArray...) = DArray(I->f(map(localpart, As)...), As[1])

@static if VERSION < v"0.6.0-dev.1632"
    for f in (:.+, :.-, :.*, :./, :.%, :.<<, :.>>, :div, :mod, :rem, :&, :|, :$)
        @eval begin
            ($f){T}(A::DArray{T}, B::Number) = map_localparts(r->($f)(r, B), A)
            ($f){T}(A::Number, B::DArray{T}) = map_localparts(r->($f)(A, r), B)
        end
    end
end

function samedist(A::DArray, B::DArray)
    (size(A) == size(B)) || throw(DimensionMismatch())
    if (procs(A) != procs(B)) || (A.cuts != B.cuts)
        B = DArray(x->B[x...], A)
    end
    B
end

for f in (:+, :-, :div, :mod, :rem, :&, :|, :$)
    @eval begin
        function ($f){T}(A::DArray{T}, B::DArray{T})
            B = samedist(A, B)
            map_localparts($f, A, B)
        end
        ($f){T}(A::DArray{T}, B::Array{T}) = map_localparts($f, A, B)
        ($f){T}(A::Array{T}, B::DArray{T}) = map_localparts($f, A, B)
    end
end
@static if VERSION < v"0.6.0-dev.1632"
    for f in (:.+, :.-, :.*, :./, :.%, :.<<, :.>>)
        @eval begin
            function ($f){T}(A::DArray{T}, B::DArray{T})
                map_localparts($f, A, B)
            end
            ($f){T}(A::DArray{T}, B::Array{T}) = map_localparts($f, A, B)
            ($f){T}(A::Array{T}, B::DArray{T}) = map_localparts($f, A, B)
        end
    end
end

function mapslices{T,N,A}(f::Function, D::DArray{T,N,A}, dims::AbstractVector)
    if !all(t -> t == 1, size(D.indexes)[dims])
        p = ones(Int, ndims(D))
        nondims = filter(t -> !(t in dims), 1:ndims(D))
        p[nondims] = defaultdist([size(D)...][[nondims...]], procs(D))
        DD = DArray(size(D), procs(D), p) do I
            return convert(A, D[I...])
        end
        return mapslices(f, DD, dims)
    end

    refs = Future[remotecall((x,y,z)->mapslices(x,localpart(y),z), p, f, D, dims) for p in procs(D)]

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
    R = Array{eltype(R1)}(Rsize...)

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

    refs = Future[remotecall((x, y, z) -> _ppeval(x, map(localpart, y)...; dim = z), p, f, D, dim) for p in procs(D[1])]

    # The array of Futures has to be reshaped for the DArray constructor to work correctly.
    # This requires a fetch and the DArray is also fetching so it might be better to modify
    # the DArray constructor.
    sd = [size(D[1].pids)...]
    nd = remotecall_fetch((r)->ndims(fetch(r)), refs[1].where, refs[1])
    DArray(reshape(refs, tuple([sd[1:nd - 1], sd[end];]...)))
end
