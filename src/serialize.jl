function Base.serialize(S::AbstractSerializer, d::DArray)
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

function Base.deserialize{T<:DArray}(S::AbstractSerializer, t::Type{T})
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

# Serialize only those parts of the object as required by the destination worker.
type PartitionedSerializer
   indexable_obj                      # An indexable object, Array, SparseMatrix, etc.
                                      # Complete object on the serializing side.
                                      # Part object on the deserialized side.
   pids::Nullable{Array}
   idxs::Nullable{Array}
   local_idxs::Nullable{Tuple}

   PartitionedSerializer(obj, local_idxs::Tuple) = new(obj, Nullable{Array}(), Nullable{Array}(), local_idxs)
   function PartitionedSerializer(obj, pids::Array, idxs::Array)
        pas = new(obj,pids,idxs,Nullable{Tuple}())

        if myid() in pids
            pas.local_idxs = idxs[findfirst(pids, myid())]
        end
        return pas
    end
end

function Base.serialize(S::AbstractSerializer, pas::PartitionedSerializer)
    pid = Base.worker_id_from_socket(S.io)
    I = get(pas.idxs)[findfirst(get(pas.pids), pid)]
    Serializer.serialize_type(S, typeof(pas))
    serialize(S, pas.indexable_obj[I...])
    serialize(S, I)
end

function Base.deserialize{T<:PartitionedSerializer}(S::AbstractSerializer, t::Type{T})
    obj_part = deserialize(S)
    I = deserialize(S)
    return PartitionedSerializer(obj_part, I)
end

function verify_and_get(pas::PartitionedSerializer, I)
    # Handle the special case where myid() is part of pas.pids.
    # For this case serialize/deserialize is not called as the remotecall is executed locally
    if myid() in get(pas.pids, [])
        @assert I == get(pas.idxs)[findfirst(get(pas.pids),myid())]
        return pas.indexable_obj[I...]
    else
        @assert I == get(pas.local_idxs, ())
        return pas.indexable_obj
    end
end
