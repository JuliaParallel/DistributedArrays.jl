# distributed ranges
type Distributed{T}
    identity::Tuple
    pids::Array{Int,1}                # pids[i]==p ⇒ processor p is has range part i
    original::Nullable{T}
    parts::Nullable{Array{T,1}}       # Only useful for UnitRange
    flags::Int

    release::Bool

    function Distributed(identity, pids, original, parts, flags)
        release = (myid() == identity[1])

        global registry
        haskey(registry, (identity, :CONTAINER)) && return registry[(identity, :CONTAINER)]

        d = new(identity, pids, original, parts, flags)
        if release
            push!(refs, identity)
            registry[(identity, :CONTAINER)] = d

            finalizer(d, close)
        end
        d
    end

    Distributed() = new()
end
export Distributed

const DISTRIBUTED_CONST = 1

function close(d::Distributed)
    if (myid() == d.identity[1]) && d.release
        @schedule close_by_identity(d.identity, d.pids)
        d.release = false
    end
    nothing
end

function distribute(u::UnitRange, pids=workers())
    parts, _cuts = chunk_idxs([length(u)], length(pids))
    parts = map(x->x[1], parts)
    identity = next_did()

    @sync for i = 1:length(pids)
        @async Base.remotecall_wait(construct_on_worker, pids[i], identity, i->parts[i], pids, u, parts, 0)
    end

    if myid() in pids
        d = registry[(identity, :CONTAINER)]
    else
        d = Distributed{UnitRange}(identity, pids, u, parts, 0)
    end
    d
end

function construct_on_worker(identity, gen_localpart, pids, original, parts, flags)
    global registry
    lp = gen_localpart(localpartindex(pids))

    registry[(identity, :LOCALPART)] = lp
    registry[(identity, :CONTAINER)] = Distributed{typeof(lp)}(identity, pids, original, parts, flags)
    nothing
end

function localpart{T}(d::Distributed{T})
    lpidx = localpartindex(d.pids)
    if lpidx == 0
        if T == UnitRange
            return (1:0)::T
        elseif (d.flags & DISTRIBUTED_CONST) == DISTRIBUTED_CONST
            return get(d.original)
        else
            error("localpart does not exist")
        end
    end

    global registry
    return registry[(d.identity, :LOCALPART)]::T
end


localpart(d::Distributed, localidx...) = localpart(d)[localidx...]

# distribute an array of objects where length(arr) == length(pids)
# localpart is an item and not an array.
function distribute_one_per_worker(x::Array, pids=workers())
    @assert length(x) == length(pids)

    identity = next_did()
    et = eltype(x)
    original = Nullable{et}()
    parts = Nullable{Array{et,1}}()

    @sync for i = 1:length(pids)
        lp_on_i = x[i]
        @async Base.remotecall_wait(construct_on_worker, pids[i], identity, i->lp_on_i, pids, original, parts, 0)
    end

    if myid() in pids
        d = registry[(identity, :CONTAINER)]
    else
        d = Distributed{et}(identity, pids, original, 0)
    end
    d
end
export distribute_one_per_worker

function distribute_one_per_worker(f::Function, pids=workers(); element_type=Any)
    identity = next_did()
    et = element_type
    original = Nullable{et}()
    parts = Nullable{Array{et,1}}()

    @sync for i = 1:length(pids)
        @async Base.remotecall_wait(construct_on_worker, pids[i], identity, p->f(p), pids, original, parts, 0)
    end

    if myid() in pids
        d = registry[(identity, :CONTAINER)]
    else
        d = Distributed{et}(identity, pids, original, parts, 0)
    end
    d
end

distribute_one_per_worker(x::Array{Future}, pids=workers()) = distribute_one_per_worker_refs(x, pids)
distribute_one_per_worker(x::Array{RemoteChannel}, pids=workers()) = distribute_one_per_worker_refs(x, pids)
function distribute_one_per_worker_refs(x, pids)
    identity = next_did()
    et=Any
    original = Nullable{et}()
    parts = Nullable{Array{et}}()

    @sync for i = 1:length(pids)
        rr = x[i]
        @async Base.remotecall_wait(construct_on_worker, pids[i], identity, i->fetch(rr), pids, original, parts, 0)
    end

    if myid() in pids
        d = registry[(identity, :CONTAINER)]
    else
        d = Distributed{et}(identity, pids, original, parts, 0)
    end
    d
end

function Base.convert{T,N,A}(Distributed, d_in::DArray{T,N,A})
    identity = next_did()
    original = Nullable{A}()
    parts = Nullable{Array{A,1}}()
    pids = reshape(d_in.pids, (length(d_in.pids),))

    @sync for i = 1:length(pids)
        @async Base.remotecall_wait(construct_on_worker, pids[i], identity, i->localpart(d_in), pids, original, parts, 0)
    end

    if myid() in pids
        d = registry[(identity, :CONTAINER)]
    else
        d = Distributed{A}(identity, pids, original, parts, 0)
    end
    d
end

function distribute_constant(x, pids=workers())
    identity = next_did()
    et = typeof(x)
    original = x
    parts = Nullable{Array{et}}()

    @sync for i = 1:length(pids)
        @async Base.remotecall_wait(construct_on_worker, pids[i], identity, i->original, pids, original, parts, DISTRIBUTED_CONST)
    end

    if myid() in pids
        d = registry[(identity, :CONTAINER)]
    else
        d = Distributed{et}(identity, pids, original, parts, DISTRIBUTED_CONST)
    end
    d
end
export distribute_constant

Base.getindex(d::Distributed) = localpart(d)

# The below will work if the local part is indexable
Base.getindex(d::Distributed, i::Int...) = localpart(d)[i...]

# get the localpart from a different worker
type DPid
    pid::Int
end
export DPid

getindex(d::Distributed, pid::DPid) = remotecall_fetch(localpart, pid.pid, d)

# mapping functions
Base.map{T<:UnitRange}(f, d::Distributed{T}) = map_impl(x->map(f,x), d)
Base.map(f, d::Distributed) = map_impl(f,d)
function map_impl(f, d::Distributed)
    refs = Array(Future, length(d.pids))
    for (i,p) in enumerate(d.pids)
        refs[i] = remotecall(d2 -> f(localpart(d2)), p, d)
    end
    distribute_one_per_worker(refs, d.pids)
end

function Base.reduce(op, d::Distributed)
    # first reduce locally and then on the caller
    results = Array(Any, length(d.pids))
    @sync for (i,p) in enumerate(d.pids)
        results[i] = remotecall_fetch(d2 -> (lp=localpart(d2); isa(lp, Array) ? reduce(op, lp) : lp), p, d)
    end
    reduce(op, results)
end

# retrieval
function Base.collect(d::Distributed; element_type=Any, flatten=true)
    res = Array(Any, length(d.pids))
    @sync for (i,p) in enumerate(d.pids)
        res[i] = d[DPid(p)]
    end
    if flatten
        return collect(element_type, Base.flatten(res))
    end
    res
end

# serializers
function Base.serialize(S::AbstractSerializer, d::Distributed)
    # Only send the ident for participating workers - we expect the DArray to exist in the
    # remote registry
    destpid = Base.worker_id_from_socket(S.io)
    Serializer.serialize_type(S, typeof(d))
    if (destpid in d.pids) || (destpid == d.identity[1])
        serialize(S, (true, d.identity))    # (identity_only, identity)
    else
        serialize(S, (false, d.identity))
        for n in [:pids, :original, :parts, :flags]
            serialize(S, getfield(d, n))
        end
    end
end

function Base.deserialize{T<:Distributed}(S::AbstractSerializer, t::Type{T})
    what = deserialize(S)
    identity_only = what[1]
    identity = what[2]

    if identity_only
        global registry
        if haskey(registry, (identity, :CONTAINER))
            return registry[(identity, :CONTAINER)]
        else
            # access to fields will throw an error, at least the deserialization process will not
            # result in worker death
            d = T()
            d.identity = identity
            return d
        end
    else
        # We are not a participating worker, deser fields and instantiate locally.
        pids = deserialize(S)
        original = deserialize(S)
        parts = deserialize(S)
        flags = deserialize(S)
        return T(identity, pids, original, parts, flags)
    end
end


