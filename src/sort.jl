# Sorting a DVector using samplesort

function sample_n_setup_ref(d::DVector, sample_size; kwargs...)
    lp = localpart(d)
    llp = length(lp)
    np = length(procs(d))
    sample_size = llp > sample_size ? sample_size : llp
    sorted = sort(lp; kwargs...)
    sample = sorted[collect(1:div(llp,sample_size):llp)]
    ref = RemoteChannel(()->Channel(np+1))             # To collect parts to be sorted locally later.
                                                       # First element is the locally sorted vector
    put!(ref, sorted)
    return (sample, ref)
end


function scatter_n_sort_localparts(d, myidx, refs, boundaries::Array{T}; by = identity, kwargs...) where T
    if d==nothing
        sorted = take!(refs[myidx])  # First entry in the remote channel is sorted localpart
    else
        sorted = sort(localpart(d); by = by, kwargs...)
    end

    # send respective parts to correct workers, iterate over sorted array
    p_sorted = 1
    for (i,r) in enumerate(refs)
        p_till = length(sorted)+1

        # calculate range to send to refs[i]
        ctr=1
        for x in sorted[p_sorted:end]
            if by(x) > by(boundaries[i+1])
                p_till = p_sorted+ctr-1
                break
            else
                ctr += 1
            end
        end

        if p_till == p_sorted
            @async put!(r, Array{T}(undef,0))
        else
            v = sorted[p_sorted:p_till-1]
            @async put!(r, v)
        end

        p_sorted = p_till
    end

    # wait to receive all of my parts from all other workers
    lp_sorting=T[]
    for _ in refs
        v = take!(refs[myidx])
        append!(lp_sorting, v)
    end

    sorted_ref=RemoteChannel()
    put!(sorted_ref, sort!(lp_sorting; by = by, kwargs...))
    return (sorted_ref, length(lp_sorting))
end

function compute_boundaries(d::DVector{T}; kwargs...) where T
    pids = procs(d)
    np = length(pids)
    sample_sz_on_wrkr = 512

    results = asyncmap(p -> remotecall_fetch(sample_n_setup_ref, p, d, sample_sz_on_wrkr; kwargs...), pids)

    samples = Array{T}(undef,0)
    for x in results
        append!(samples, x[1])
    end
    sort!(samples; kwargs...)
    samples[1] = typemin(T)

    refs=[x[2] for x in results]

    boundaries = samples[[1+(x-1)*div(length(samples), np) for x in 1:np]]
    push!(boundaries, typemax(T))

    return (boundaries, refs)
end

"""
    sort(d::DVector; sample=true, kwargs...) -> DVector

Sorts and returns a new distributed vector.

The sorted vector may not have the same distribution as the original.

Keyword argument `sample` can take values:

- `true`: A sample of max size 512 is first taken from all nodes. This is used to balance the distribution of the sorted array on participating workers. Default is `true`.

- `false`: No sampling is done. Assumes a uniform distribution between min(d) and max(d)

- 2-element tuple of the form `(min, max)`: No sampling is done. Assumes a uniform distribution between specified min and max values

- Array{T}: The passed array is assumed to be a sample of the distribution and is used to balance the sorted distribution.

Keyword argument `alg` takes the same options `Base.sort`
"""
function Base.sort(d::DVector{T}; sample=true, kwargs...) where T
    pids = procs(d)
    np = length(pids)

    # Only `alg` and `sample` are supported as keyword arguments
    if length(filter(x->!(x in (:alg, :by)), [x[1] for x in kwargs])) > 0
        throw(ArgumentError("Only `alg`, `by` and `sample` are supported as keyword arguments"))
    end

    if sample==true
        boundaries, refs = compute_boundaries(d; kwargs...)
        presorted=true

    elseif sample==false
        # Assume an uniform distribution between min and max values
        minmax=asyncmap(p->remotecall_fetch(d->(minimum(localpart(d)), maximum(localpart(d))), p, d), pids)
        min_d = minimum(T[x[1] for x in minmax])
        max_d = maximum(T[x[2] for x in minmax])

        return sort(d; sample=(min_d,max_d), kwargs...)

    elseif isa(sample, Tuple)
        # Assume an uniform distribution between min and max values in the tuple
        lb=sample[1]
        ub=sample[2]

        @assert lb<=ub

        s = Array{T}(undef,np)
        part = abs(ub - lb)/np
        (isnan(part) || isinf(part)) && throw(ArgumentError("lower and upper bounds must not be infinities"))

        for n in 1:np
            v = lb + (n-1)*part
            if T <: Integer
                s[n] = round(v)
            else
                s[n] = v
            end
        end
        return sort(d; sample=s, kwargs...)

    elseif isa(sample, Array)
        # Provided array is used as a sample
        samples = sort(copy(sample))
        samples[1] = typemin(T)
        boundaries = samples[[1+(x-1)*div(length(samples), np) for x in 1:np]]
        push!(boundaries, typemax(T))
        presorted=false

        refs=[RemoteChannel(p) for p in procs(d)]
    else
        throw(ArgumentError("keyword arg `sample` must be Boolean, Tuple(Min,Max) or an actual sample of data : " * string(sample)))
    end

    local_sort_results = Array{Tuple}(undef,np)

    Base.asyncmap!((i,p) -> remotecall_fetch(
            scatter_n_sort_localparts, p, presorted ? nothing : d, i, refs, boundaries; kwargs...),
                                    local_sort_results, 1:np, pids)

    # Construct a new DArray from the sorted refs. Remove parts with 0-length since
    # the DArray constructor_from_refs does not yet support it. This implies that
    # the participating workers for the sorted darray may be different from the original
    # for highly non-uniform distributions.
    local_sorted_refs = RemoteChannel[x[1] for x in filter(x->x[2]>0, local_sort_results)]
    return DArray(local_sorted_refs)
end
