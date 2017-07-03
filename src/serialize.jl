function Base.serialize{T,N,A}(S::AbstractSerializer, d::DArray{T,N,A})
    # Only send the ident for participating workers - we expect the DArray to exist in the
    # remote registry. DO NOT send the localpart.
    destpid = Base.worker_id_from_socket(S.io)
    Serializer.serialize_type(S, typeof(d))
    if (destpid in d.pids) || (destpid == d.id[1])
        serialize(S, (true, d.id))    # (id_only, id)
    else
        serialize(S, (false, d.id))
        for n in [:dims, :pids, :indexes, :cuts]
            serialize(S, getfield(d, n))
        end
        serialize(S, A)
    end
end

function Base.deserialize{DT<:DArray}(S::AbstractSerializer, t::Type{DT})
    what = deserialize(S)
    id_only = what[1]
    id = what[2]

    if id_only
        d = d_from_weakref_or_d(id)
        if d === nothing
            # access to fields will throw an error, at least the deserialization process will not
            # result in worker death
            d = DT()
            d.id = id
        end
        return d
    else
        # We are not a participating worker, deser fields and instantiate locally.
        dims = deserialize(S)
        pids = deserialize(S)
        indexes = deserialize(S)
        cuts = deserialize(S)
        A = deserialize(S)
        T=eltype(DT)
        N=length(dims)
        return DT(id, dims, pids, indexes, cuts, empty_localpart(T,N,A))
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
