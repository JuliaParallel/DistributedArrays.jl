using DistributedArrays, Test
import ExplicitImports

@testset "ExplicitImports" begin
    # No implicit imports in DistributedArrays (ie. no `using MyPkg`)
    @test ExplicitImports.check_no_implicit_imports(DistributedArrays) === nothing

    # No non-owning imports in DistributedArrays (ie. no `using LinearAlgebra: map`)
    @test ExplicitImports.check_all_explicit_imports_via_owners(DistributedArrays) === nothing

    # Limit non-public imports in DistributedArrays (ie. `using MyPkg: _non_public_internal_func`)
    # to a few selected types and functions
    @test ExplicitImports.check_all_explicit_imports_are_public(
        DistributedArrays;
        ignore = (
            # Base
            :Broadcasted,
            :Callable,
            (VERSION < v"1.11" ? (:tail,) : ())...,
        ),
    ) === nothing

    # No stale imports in DistributedArrays (ie. no `using MyPkg: func` where `func` is not used in DistributedArrays)
    @test ExplicitImports.check_no_stale_explicit_imports(DistributedArrays) === nothing

    # No non-owning accesses in DistributedArrays (ie. no `... LinearAlgebra.map(...)`)
    @test ExplicitImports.check_all_qualified_accesses_via_owners(DistributedArrays) === nothing

    # Limit non-public accesses in DistributedArrays (ie. no `... MyPkg._non_public_internal_func(...)`)
    # to a few selected types and methods from Base
    @test ExplicitImports.check_all_qualified_accesses_are_public(
        DistributedArrays;
        ignore = (
            # Base.Broadcast
            :AbstractArrayStyle,
            :DefaultArrayStyle,
            :broadcasted,
            :throwdm,
            # Base
            (VERSION < v"1.11" ? (Symbol("@propagate_inbounds"),) : ())...,
            :ReshapedArray,
            :Slice,
            :_all,
            :_any,
            :_mapreduce,
            :check_reducedims,
            :checkbounds_indices,
            :index_lengths,
            :mapreducedim!,
            :promote_op,
            :reducedim_initarray,
            :reindex,
            :setindex_shape_check,
            :unalias,
            # Serialization
            :serialize_type,
            # Statistics        
            :_mean,
        ),
    ) === nothing

    # No self-qualified accesses in DistributedArrays (ie. no `... DistributedArrays.func(...)`)
    @test ExplicitImports.check_no_self_qualified_accesses(DistributedArrays) === nothing
end
