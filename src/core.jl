# Thread-safe registry of DArray references
struct DArrayRegistry
    data::Dict{Tuple{Int,Int}, Any}
    lock::ReentrantLock
    DArrayRegistry() = new(Dict{Tuple{Int,Int}, Any}(), ReentrantLock())
end
const REGISTRY = DArrayRegistry()

function Base.get(r::DArrayRegistry, id::Tuple{Int,Int}, default)
    @lock r.lock begin
        return get(r.data, id, default)
    end
end
function Base.getindex(r::DArrayRegistry, id::Tuple{Int,Int})
    @lock r.lock begin
        return r.data[id]
    end
end
function Base.setindex!(r::DArrayRegistry, val, id::Tuple{Int,Int})
    @lock r.lock begin
        r.data[id] = val
    end
    return r
end
function Base.delete!(r::DArrayRegistry, id::Tuple{Int,Int})
    @lock r.lock delete!(r.data, id)
    return r
end

# Thread-safe set of IDs of DArrays created on this node
struct DArrayRefs
    data::Set{Tuple{Int,Int}}
    lock::ReentrantLock
    DArrayRefs() = new(Set{Tuple{Int,Int}}(), ReentrantLock())
end
const REFS = DArrayRefs()

function Base.push!(r::DArrayRefs, id::Tuple{Int,Int})
    # Ensure id refers to a DArray created on this node
    if first(id) != myid()
        throw(
            ArgumentError(
                lazy"`DArray` is not created on the current worker: Only `DArray`s created on worker $(myid()) can be stored in this set but the `DArray` was created on worker $(first(id))."))
    end
    @lock r.lock begin
        return push!(r.data, id)
    end
end
function Base.delete!(r::DArrayRefs, id::Tuple{Int,Int})
    @lock r.lock delete!(r.data, id)
    return r
end

# Global counter to generate a unique ID for each DArray
const DID = Threads.Atomic{Int}(1)

"""
    next_did()

Increment a global counter and return a tuple of the current worker ID and the incremented
value of the counter.

This tuple is used as a unique ID for a new `DArray`.
"""
next_did() = (myid(), Threads.atomic_add!(DID, 1))

release_localpart(id::Tuple{Int,Int}) = (delete!(REGISTRY, id); nothing)
function release_allparts(id::Tuple{Int,Int}, pids::Array{Int})
    @sync begin
        released_myid = false
        for p in pids
            if p == myid()
                @async release_localpart(id)
                released_myid = true
            else
                @async remotecall_fetch(release_localpart, p, id)
            end
        end
        if !released_myid
            @async release_localpart(id)
        end
    end
    return nothing
end

function close_by_id(id::Tuple{Int,Int}, pids::Array{Int})
    release_allparts(id, pids)
    delete!(REFS, id)
    nothing
end

function d_closeall()
    @lock REFS.lock begin
        while !isempty(REFS.data)
            id = pop!(REFS.data)
            d = d_from_weakref_or_d(id)
            if d isa DArray
                finalize(d)
            end
        end
    end
    return nothing
end

Base.close(d::DArray) = finalize(d)

"""
    procs(d::DArray)

Get the vector of processes storing pieces of DArray `d`.
"""
Distributed.procs(d::DArray)    = d.pids
Distributed.procs(d::SubDArray) = procs(parent(d))

"""
    localpart(A)

The identity when input is not distributed
"""
localpart(A) = A
