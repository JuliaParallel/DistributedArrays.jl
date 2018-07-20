const registry=Dict{Tuple, Any}()
const refs=Set()  # Collection of darray identities created on this node

let DID::Int = 1
    global next_did
    next_did() = (id = DID; DID += 1; (myid(), id))
end

"""
    next_did()

Produces an incrementing ID that will be used for DArrays.
"""
next_did

release_localpart(id::Tuple) = (delete!(registry, id); nothing)
release_localpart(d) = release_localpart(d.id)

function close_by_id(id, pids)
#   @async println("Finalizer for : ", id)
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

function Base.close(d::DArray)
#    @async println("close : ", d.id, ", object_id : ", object_id(d), ", myid : ", myid() )
    if (myid() == d.id[1]) && d.release
        @async close_by_id(d.id, d.pids)
        d.release = false
    end
    nothing
end

function d_closeall()
    crefs = copy(refs)
    for id in crefs
        if id[1] ==  myid() # sanity check
            haskey(registry, id) && close(d_from_weakref_or_d(id))
            yield()
        end
    end
end

"""
    procs(d::DArray)

Get the vector of processes storing pieces of DArray `d`.
"""
Distributed.procs(d::DArray) = d.pids

"""
    localpart(A)

The identity when input is not distributed
"""
localpart(A) = A

