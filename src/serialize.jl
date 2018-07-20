function Serialization.serialize(S::AbstractSerializer, d::DArray{T,N,A}) where {T,N,A}
    # Only send the ident for participating workers - we expect the DArray to exist in the
    # remote registry. DO NOT send the localpart.
    destpid = Distributed.worker_id_from_socket(S.io)
    Serializer.serialize_type(S, typeof(d))
    if (destpid in d.pids) || (destpid == d.id[1])
        serialize(S, (true, d.id))    # (id_only, id)
    else
        serialize(S, (false, d.id))
        for n in [:dims, :pids, :indices, :cuts]
            serialize(S, getfield(d, n))
        end
        serialize(S, A)
    end
end

function Serialization.deserialize(S::AbstractSerializer, t::Type{DT}) where DT<:DArray
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
        indices = deserialize(S)
        cuts = deserialize(S)
        A = deserialize(S)
        T=eltype(DT)
        N=length(dims)
        return DT(id, dims, pids, indices, cuts, empty_localpart(T,N,A))
    end
end

# Serialize only those parts of the object as required by the destination worker.
mutable struct DestinationSerializer
    generate::Union{Function,Missing}     # Function to generate the part to be serialized
    pids::Union{Array,Missing}            # MUST have the same shape as the distribution

    deser_obj::Union{Any,Missing}         # Deserialized part

    DestinationSerializer(f,p,d) = new(f,p,d)
end

DestinationSerializer(f::Function, pids::Array) = DestinationSerializer(f, pids, missing)

# contructs a DestinationSerializer after verifying that the shape of pids.
function verified_destination_serializer(f::Function, pids::Array, verify_size)
    @assert size(pids) == verify_size
    return DestinationSerializer(f, pids)
end

DestinationSerializer(deser_obj::Any) = DestinationSerializer(missing, missing, deser_obj)

function Serialization.serialize(S::AbstractSerializer, s::DestinationSerializer)
    pid = Distributed.worker_id_from_socket(S.io)
    pididx = findfirst(isequal(pid), s.pids)
    Serialization.serialize_type(S, typeof(s))
    serialize(S, s.generate(pididx))
end

function Serialization.deserialize(S::AbstractSerializer, t::Type{T}) where T<:DestinationSerializer
    lpart = deserialize(S)
    return DestinationSerializer(lpart)
end


function localpart(s::DestinationSerializer)
    if !ismissing(s.deser_obj)
        return s.deser_obj
    elseif  !ismissing(s.generate) && (myid() in s.pids)
        # Handle the special case where myid() is part of s.pids.
        # In this case serialize/deserialize is not called as the remotecall is executed locally
        return s.generate(findfirst(isequal(myid()), s.pids))
    else
        throw(ErrorException(string("Invalid state in DestinationSerializer.")))
    end
end
