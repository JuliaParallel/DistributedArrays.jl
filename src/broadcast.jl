###
# Distributed broadcast implementation
##

# We define a custom ArrayStyle here since we need to keep track of
# the fact that it is Distributed and what kind of underlying broadcast behaviour
# we will encounter.
struct DArrayStyle{Style <: BroadcastStyle} <: Broadcast.AbstractArrayStyle{Any} end
DArrayStyle(::S) where {S} = DArrayStyle{S}()
DArrayStyle(::S, ::Val{N}) where {S,N} = DArrayStyle(S(Val(N)))
DArrayStyle(::Val{N}) where N = DArrayStyle{Broadcast.DefaultArrayStyle{N}}()

Broadcast.BroadcastStyle(::Type{<:DArray{<:Any, N, A}}) where {N, A} = DArrayStyle(BroadcastStyle(A), Val(N))

# promotion rules
# TODO: test this
function Broadcast.BroadcastStyle(::DArrayStyle{AStyle}, ::DArrayStyle{BStyle}) where {AStyle, BStyle}
    DArrayStyle(BroadcastStyle(AStyle, BStyle))
end

function Broadcast.broadcasted(::DArrayStyle{Style}, f, args...) where Style
    inner = Broadcast.broadcasted(Style(), f, args...)
    if inner isa Broadcasted
        return Broadcasted{DArrayStyle{Style}}(inner.f, inner.args, inner.axes)
    else # eagerly evaluated
        return inner
    end
end

# # deal with one layer deep lazy arrays
# BroadcastStyle(::Type{<:LinearAlgebra.Transpose{<:Any,T}}) where T <: DArray = BroadcastStyle(T)
# BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{<:Any,T}}) where T <: DArray = BroadcastStyle(T)
# BroadcastStyle(::Type{<:SubArray{<:Any,<:Any,<:T}}) where T <: DArray = BroadcastStyle(T)

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
# DArray across workers is equal or that the underlying array type is consistent.
#
# Implementation:
#   - first distribute all arguments
#     - Q: How do decide on the cuts
#   - then localise arguments on each node
##
@inline function Base.copyto!(dest::DDestArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))

    # Distribute Broadcasted
    # This will turn local AbstractArrays into DArrays
    dbc = bcdistribute(bc)

    @sync for p in procs(dest)
        @async remotecall_wait(p) do
            # get the indices for the localpart
            lpidx = localpartindex(dest)
            @assert lpidx != 0
            # create a local version of the broadcast, by constructing views
            # Note: creates copies of the argument
            lbc = bclocal(dbc, dest.indices[lpidx])
            copyto!(localpart(dest), lbc)
        end
    end

    return dest
end

# Test
# a = Array
# a .= DArray(x,y)

@inline function Base.copy(bc::Broadcasted{<:DArrayStyle})
    dbc = bcdistribute(bc)
    # TODO: teach DArray about axes since this is wrong for OffsetArrays
    DArray(map(length, axes(bc))) do I
        lbc = bclocal(dbc, I)
        copy(lbc)
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
_bcdistribute(::Any, x) = x

@inline bcdistribute_args(args::Tuple) = (bcdistribute(args[1]), bcdistribute_args(tail(args))...)
bcdistribute_args(args::Tuple{Any}) = (bcdistribute(args[1]),)
bcdistribute_args(args::Tuple{}) = ()

# dropping axes here since recomputing is easier
@inline bclocal(bc::Broadcasted{DArrayStyle{Style}}, idxs) where Style = Broadcasted{Style}(bc.f, bclocal_args(_bcview(axes(bc), idxs), bc.args))

# bclocal will do a view of the data and the copy it over
# except when the data already is local
function bclocal(x::DArray{T, N, AT}, idxs) where {T, N, AT}
    bcidxs = _bcview(axes(x), idxs)
    makelocal(x, bcidxs...)
end
bclocal(x, idxs) = x

@inline bclocal_args(idxs, args::Tuple) = (bclocal(args[1], idxs), bclocal_args(idxs, tail(args))...)
bclocal_args(idxs, args::Tuple{Any}) = (bclocal(args[1], idxs),)
bclocal_args(idxs, args::Tuple{}) = ()
