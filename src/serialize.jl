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
type DestinationSerializer
    generate::Nullable{Function}     # Function to generate the part to be serialized
    pids::Nullable{Array}            # MUST have the same shape as the distribution

    deser_obj::Nullable{Any}         # Deserialized part

    DestinationSerializer(f,p,d) = new(f,p,d)
end

DestinationSerializer(f::Function, pids::Array) = DestinationSerializer(f, pids, Nullable{Any}())

# contructs a DestinationSerializer after verifying that the shape of pids.
function verified_destination_serializer(f::Function, pids::Array, verify_size)
    @assert size(pids) == verify_size
    return DestinationSerializer(f, pids)
end

DestinationSerializer(deser_obj::Any) = DestinationSerializer(Nullable{Function}(), Nullable{Array}(), deser_obj)

function Base.serialize(S::AbstractSerializer, s::DestinationSerializer)
    pid = Base.worker_id_from_socket(S.io)
    pididx = findfirst(get(s.pids), pid)
    Serializer.serialize_type(S, typeof(s))
    serialize(S, get(s.generate)(pididx))
end

function Base.deserialize{T<:DestinationSerializer}(S::AbstractSerializer, t::Type{T})
    lpart = deserialize(S)
    return DestinationSerializer(lpart)
end


function localpart(s::DestinationSerializer)
    if !isnull(s.deser_obj)
        return get(s.deser_obj)
    elseif  !isnull(s.generate) && (myid() in get(s.pids))
        # Handle the special case where myid() is part of s.pids.
        # In this case serialize/deserialize is not called as the remotecall is executed locally
        return get(s.generate)(findfirst(get(s.pids), myid()))
    else
        throw(ErrorException(string("Invalid state in DestinationSerializer.")))
    end
end
