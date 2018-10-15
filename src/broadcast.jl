###
# Distributed broadcast implementation
##

using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted, _max

# We define a custom ArrayStyle here since we need to keep track of
# the fact that it is Distributed and what kind of underlying broadcast behaviour
# we will encounter.
struct DArrayStyle{Style <: BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
DArrayStyle(::S) where {S} = DArrayStyle{S}()
DArrayStyle(::S, ::Val{N}) where {S,N} = DArrayStyle(S(Val(N)))
DArrayStyle(::Val{N}) where N = DArrayStyle{Broadcast.DefaultArrayStyle{N}}()

BroadcastStyle(::Type{<:DArray{<:Any, N, A}}) where {N, A} = DArrayStyle(BroadcastStyle(A), Val(N))

# promotion rules
function BroadcastStyle(::DArrayStyle{AStyle}, ::DArrayStyle{BStyle}) where {AStyle, BStyle}
    DArrayStyle{BroadcastStyle(AStyle, BStyle)}()
end

# # deal with one layer deep lazy arrays
# BroadcastStyle(::Type{<:LinearAlgebra.Transpose{<:Any,T}}) where T <: DArray = BroadcastStyle(T)
# BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{<:Any,T}}) where T <: DArray = BroadcastStyle(T)
BroadcastStyle(::Type{<:SubArray{<:Any,<:Any,<:T}}) where T <: DArray = BroadcastStyle(T)

# # This Union is a hack. Ideally Base would have a Transpose <: WrappedArray <: AbstractArray
# # and we could define our methods in terms of Union{DArray, WrappedArray{<:Any, <:DArray}}
# const DDestArray = Union{DArray,
#                          LinearAlgebra.Transpose{<:Any,<:DArray},
#                          LinearAlgebra.Adjoint{<:Any,<:DArray},
#                          SubArray{<:Any, <:Any, <:DArray}}
const DDestArray = DArray

# This method is responsible for selection the output type of broadcast
function Base.similar(bc::Broadcasted{<:DArrayStyle{Style}}, ::Type{ElType}) where {Style, ElType}
    DArray(map(length, axes(bc))) do I 
        # create fake Broadcasted for underlying ArrayStyle
        bc′ = Broadcasted{Style}(identity, (), map(length, I))
        similar(bc′, ElType)
    end
end

##
# We purposefully only specialise `copyto!`,
# Broadcast implementation that defers to the underlying BroadcastStyle. We can't 
# assume that `getindex` is fast, furthermore  we can't assume that the distribution of
# DArray accross workers is equal or that the underlying array type is consistent.
#
# Implementation:
#   - first distribute all arguments
#     - Q: How do decide on the cuts
#   - then localise arguments on each node
##
@inline function Base.copyto!(dest::DDestArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))

    # Distribute Broadcasted
    # This will turn local AbstractArrays into DArrays
    dbc = bcdistribute(bc)

    asyncmap(procs(dest)) do p
        remotecall_fetch(p) do
            # get the indices for the localpart
            lidcs = localindices(dest)
            # create a local version of the broadcast, by constructing views
            # Note: creates copies of the argument
            lbc = bclocal(dbc, lidcs)
            Base.copyto!(localpart(dest), lbc)
            return nothing
        end
    end
    return dest
end

function _linear(shape, cI)
    Is = LinearIndices(shape)[cI...]
    minI = minimum(Is)
    maxI = maximum(Is)
    if length(Is) > 2
        stride = Is[2] - Is[1]
    else
        stride = 1
    end

    if stride > 1
        I = minI:stride:maxI
    else 
        I = minI:maxI
    end

    # Sanity check
    _I = LinearIndices(I)
    if length(Is) == length(_I) && all(i1 == i2 for (i1, i2) in zip(Is, _I))
        return I
    else
        @error "_linear failed for" shape cI I
        error("Can't convert cartesian index to linear")
    end
end

function _cartesian(shape, linearI::AbstractUnitRange)
    if length(linearI) == 1
        # Ok this is really weird, we can't do the bwloe since that will
        # lead to a single CartesianIndex which will makes us really sad
        # we can't fall back to _cartesian(shape, linearI) since that will
        # give us view of 0-dimension.
        cI = _cartesian(shape, first(linearI))[1]
        n = length(cI)
        cI = ntuple(n) do i
           if i == n
               return cI[i]:cI[i]
           else 
              return i
           end
        end
        return cI
    end
    Is = CartesianIndices(shape)[linearI]
    minI = minimum(Is)
    maxI = maximum(Is)

    I = ntuple(length(minI)) do i
	ci = minI[i]:maxI[i]
        if length(ci) == 1
            return first(ci)
        else
            return ci
        end
    end
    # Sanity check
    _I = CartesianIndices(I)
    if length(Is) == length(_I) && all(i1 == i2 for (i1, i2) in zip(Is, _I))
        return I
    else
        @error "_cartesian failed for" shape linearI I
        error("Can't create cartesian index from linear index")
    end
end

function _cartesian(shape, linearI::Integer)
    I = (CartesianIndices(shape)[linearI], )
    return I
end

function Broadcast.dotview(D::DArray, args...)
    if length(args) == 1  && length(args) != ndims(D) && args[1] isa UnitRange
        return view(D, _cartesian(size(D), args[1])...)
    end
    return Base.maybeview(D, args...)
end

@inline function Base.copyto!(dest::SubDArray, bc::Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    dbc = bcdistribute(bc)

    asyncmap(procs(dest)) do p
        remotecall_fetch(p) do
            # check if we are holding part of dest, and which part
            lidcs = localindices(parent(dest)) 
            I = map(intersect, dest.indices, lidcs)
            any(isempty, I) && return nothing

            # check if the part we are holding is part of dbc
            # this should always be true...
            if length(I) == length(axes(dbc))
                any(isempty, map(intersect, axes(dbc), I)) && return nothing
                bcI = I
		lviewidcs = tolocalindices(lidcs, I)
            elseif length(I) > length(axes(dbc)) && length(axes(dbc)) == 1
                # project the axes of dbc to cartesian indices in dest
                # this can happen due to the dotview optimisation of avoiding ReshaphedArray
	        ax = _cartesian(size(parent(dest)), axes(dbc)[1])
                any(isempty, map(intersect, ax, I)) && return nothing
                bcI = (_linear(axes(parent(dest)), I), )
		lviewidcs = (_linear(axes(localpart(parent(dest))), tolocalindices(lidcs, I)),)
            else
                @assert "$(I) and $(axes(dbc)) are not compatible"
            end
            # if we gotten into the second case I is not correct for bclocal
	    lbc = bclocal(dbc, bcI)
            lbc = Broadcast.instantiate(lbc)

            Base.copyto!(view(localpart(parent(dest)), lviewidcs...), lbc)
            return nothing
	end
    end
    return dest
end

@inline function Base.copy(bc::Broadcasted{<:DArrayStyle})
    dbc = bcdistribute(bc)
    # TODO: teach DArray about axes since this is wrong for OffsetArrays
    DArray(map(length, axes(bc))) do I
        lbc = Broadcast.instantiate(bclocal(dbc, I))
        return copy(lbc)
    end
end

# _bcview creates takes the shapes of a view and the shape of a broadcasted argument,
# and produces the view over that argument that constitutes part of the broadcast
# it is in a sense the inverse of _bcs in Base.Broadcast
_bcview(::Tuple{}, ::Tuple{}) = ()
_bcview(::Tuple{}, view::Tuple) = ()
_bcview(shape::Tuple, ::Tuple{}) = (shape[1], _bcview(tail(shape), ())...)
function _bcview(shape::Tuple, view::Tuple)
    return (_bcview1(shape[1], view[1]), _bcview(tail(shape), tail(view))...)
end

# _bcview1 handles the logic for a single dimension
function _bcview1(a, b)
    if a == 1 || a == 1:1
        return 1:1
    elseif first(a) <= first(b) <= last(a) &&
           first(a) <= last(b)  <= last(b)
        return b
    else
        throw(DimensionMismatch("broadcast view could not be constructed"))
    end
end

# Distribute broadcast
# TODO: How to decide on cuts
@inline bcdistribute(bc::Broadcasted{Style}) where Style = Broadcasted{DArrayStyle{Style}}(bc.f, bcdistribute_args(bc.args), bc.axes)
@inline bcdistribute(bc::Broadcasted{Style}) where Style<:DArrayStyle = Broadcasted{Style}(bc.f, bcdistribute_args(bc.args), bc.axes)

# ask BroadcastStyle to decide if argument is in need of being distributed
bcdistribute(x::T) where T = _bcdistribute(BroadcastStyle(T), x)
_bcdistribute(::DArrayStyle, x) = x
# Don't bother distributing singletons
_bcdistribute(::Broadcast.AbstractArrayStyle{0}, x) = x
_bcdistribute(::Broadcast.AbstractArrayStyle, x) = distribute(x)
_bcdistribute(::Broadcast.AbstractArrayStyle, x::AbstractRange) = x
_bcdistribute(::Any, x) = x

@inline bcdistribute_args(args::Tuple) = (bcdistribute(args[1]), bcdistribute_args(tail(args))...)
bcdistribute_args(args::Tuple{Any}) = (bcdistribute(args[1]),)
bcdistribute_args(args::Tuple{}) = ()

# dropping axes here since recomputing is easier
@inline bclocal(bc::Broadcasted{DArrayStyle{Style}}, idxs) where Style = Broadcasted{Style}(bc.f, bclocal_args(_bcview(axes(bc), idxs), bc.args))

bclocal(x::T, idxs) where T = _bclocal(BroadcastStyle(T), x, idxs)

# bclocal will do a view of the data and the copy it over
# except when the data already is local
function _bclocal(::DArrayStyle, x, idxs)
    bcidxs = _bcview(axes(x), idxs)
    makelocal(x, bcidxs...)
end
_bclocal(::Broadcast.AbstractArrayStyle{0}, x, idxs) = x
function _bclocal(::Broadcast.AbstractArrayStyle, x::AbstractRange, idxs)
    @assert length(idxs) == 1
    x[idxs[1]]
end
function _bclocal(::Broadcast.Style{Tuple}, x, idxs)
    @assert length(idxs) == 1
    tuple((e for (i,e) in enumerate(x) if i in idxs[1])...)
end
_bclocal(::Any, x, idxs) = error("don't know how to localise $x with $idxs")

@inline bclocal_args(idxs, args::Tuple) = (bclocal(args[1], idxs), bclocal_args(idxs, tail(args))...)
bclocal_args(idxs, args::Tuple{Any}) = (bclocal(args[1], idxs),)
bclocal_args(idxs, args::Tuple{}) = ()
