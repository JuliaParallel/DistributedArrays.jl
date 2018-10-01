using Test, LinearAlgebra, SpecialFunctions
using Statistics: mean
using SparseArrays: nnz
using Random
@everywhere using SparseArrays: sprandn

@testset "test distribute and other constructors" begin
    A = rand(1:100, (100,100))

    @testset "test default distribute" begin
        DA = distribute(A)
        @test length(procs(DA)) == nworkers()
        @test sum(DA) == sum(A)
        close(DA)
    end

    @testset "test distribute with procs arguments" begin
        DA = distribute(A, procs = procs())
        @test length(procs(DA)) == nprocs()
        @test sum(DA) == sum(A)
        close(DA)
    end

    @testset "test distribute with procs and dist arguments" begin
        DA = distribute(A, procs = [1, 2], dist = [1,2])
        @test size(procs(DA)) == (1,2)
        @test sum(DA) == sum(A)
        close(DA)
    end

    @testset "Create darray with unconventional distribution and distibute like it" begin
        block = 10
        Y = nworkers() * block
        X = nworkers() * block
        remote_parts = map(workers()) do wid
            remotecall(rand, wid, block, Y)
        end
        DA1 = DArray(reshape(remote_parts, (length(remote_parts), 1)))
        A = rand(X, Y)
        DA2 = distribute(A, DA1)

        @test size(DA1) == size(DA2)

        close(DA1)
        close(DA2)
    end

    @testset "Global DArray serialization issue #134" begin
        global A134 = drandn(1)
        D2 = DArray(I -> DistributedArrays.localpart(A134), A134)
        @test D2 == A134
        close(A134)
        close(D2)
    end

    @testset "empty_localpart should work when only constructor (not conversion is defined)" begin
        @test DistributedArrays.empty_localpart(Float64,2,LowerTriangular{Float64,Matrix{Float64}}) isa
                LowerTriangular
    end
    
    @testset "Consistent Uneven Distribution issue #166" begin
        DA = drand((2+length(OTHERIDS),), [MYID, OTHERIDS])
        @test fetch(@spawnat MYID length(localpart(DA)) == 2)
        @test fetch(@spawnat OTHERIDS length(localpart(DA)) == 1)
        close(DA)
    end
end

check_leaks()

@testset "test DArray equality/copy/deepcopy" begin
    D = drand((200,200), [MYID, OTHERIDS])

    @testset "test isequal(::DArray, ::DArray)" begin
        DC = copy(D)
        @test D == DC
        close(DC)
    end

    @testset "test [deep]copy(::DArray) does a copy of each localpart" begin
        DC = copy(D)
        @spawnat OTHERIDS localpart(DC)[1] = 0
        @test fetch(@spawnat OTHERIDS localpart(D)[1] != 0)
        DD = deepcopy(D)
        @spawnat OTHERIDS localpart(DD)[1] = 0
        @test fetch(@spawnat OTHERIDS localpart(D)[1] != 0)
        close(DC)
        close(DD)
    end

    @testset "test copy(::DArray) is shallow" begin
        DA = @DArray [rand(100) for i=1:10]
        DC = copy(DA)
        id = procs(DC)[1]
        @test DA == DC
        fetch(@spawnat id localpart(DC)[1] .= -1.0)
        @test DA == DC
        @test fetch(@spawnat id all(localpart(DA)[1] .== -1.0))
        close(DA)
        close(DC)
    end

    @testset "test deepcopy(::DArray) is not shallow" begin
        DA = @DArray [rand(100) for i=1:10]
        DC = deepcopy(DA)
        id = procs(DC)[1]
        @test DA == DC
        fetch(@spawnat id localpart(DC)[1] .= -1.0)
        @test DA != DC
        @test fetch(@spawnat id all(localpart(DA)[1] .>= 0.0))
        close(DA)
        close(DC)
    end

    close(D)
end

check_leaks()

@testset "test DArray similar" begin
    D = drand((200,200), [MYID, OTHERIDS])
    DS = similar(D,Float16)

    @testset "test eltype of a similar" begin
        @test eltype(DS) == Float16
    end

    @testset "test dims of a similar" begin
        @test size(D) == size(DS)
    end
    close(D)
    close(DS)
end

check_leaks()

@testset "test DArray reshape" begin
    D = drand((200,200), [MYID, OTHERIDS])

    @testset "Test error-throwing in reshape" begin
        @test_throws DimensionMismatch reshape(D,(100,100))
    end

    DR = reshape(D,(100,400))
    @testset "Test reshape" begin
        @test size(DR) == (100,400)
    end
    close(D)
end

check_leaks()

@testset "test @DArray comprehension constructor" begin

    @testset "test valid use of @DArray" begin
        D = @DArray [i+j for i=1:10, j=1:10]
        @test D == [i+j for i=1:10, j=1:10]
        close(D)
    end

    @testset "test invalid use of @DArray" begin
        #@test_throws ArgumentError eval(:((@DArray [1,2,3,4])))
        @test_throws LoadError eval(:((@DArray [1,2,3,4])))
    end
end

check_leaks()

@testset "test DArray / Array conversion" begin
    D = drand((200,200), [MYID, OTHERIDS])

    @testset "test construct Array from (Sub)DArray" begin
        S = Matrix{Float64}(D[1:150, 1:150])
        A = Matrix{Float64}(D)

        @test A[1:150,1:150] == S
        D2 = DArray{Float64,2,Matrix{Float64}}(A)
        @test D2 == D
        DistributedArrays.allowscalar(true)
        @test fetch(@spawnat MYID localpart(D)[1,1]) == D[1,1]
        @test fetch(@spawnat OTHERIDS localpart(D)[1,1]) == D[1,101]
        DistributedArrays.allowscalar(false)
        close(D2)

        S2 = Vector{Float64}(D[4, 23:176])
        @test A[4, 23:176] == S2

        S3 = Vector{Float64}(D[23:176, 197])
        @test A[23:176, 197] == S3

        S4 = zeros(4)
        setindex!(S4, D[3:4, 99:100], :)
        # FixMe! Hitting the AbstractArray fallback here is extremely unfortunate but vec() becomes a ReshapedArray which makes it diffuclt to hit DArray methods. Unless this can be fixed in Base, we might have to add special methods for ReshapedArray{DArray}
        DistributedArrays.allowscalar(true)
        @test S4 == vec(D[3:4, 99:100])
        @test S4 == vec(A[3:4, 99:100])
        DistributedArrays.allowscalar(false)

        S5 = zeros(2,2)
        setindex!(S5, D[1,1:4], :, 1:2)
        # FixMe! Hitting the AbstractArray fallback here is extremely unfortunate but vec() becomes a ReshapedArray which makes it diffuclt to hit DArray methods. Unless this can be fixed in Base, we might have to add special methods for ReshapedArray{DArray}
        DistributedArrays.allowscalar(true)
        @test vec(S5) == D[1, 1:4]
        @test vec(S5) == A[1, 1:4]
        DistributedArrays.allowscalar(false)
    end
    close(D)
end

check_leaks()

@testset "test copy!" begin
    D1 = dzeros((10,10))
    r1 = remotecall_wait(() -> randn(3,10), workers()[1])
    r2 = remotecall_wait(() -> randn(7,10), workers()[2])
    D2 = DArray(reshape([r1; r2], 2, 1))
    copyto!(D2, D1)
    @test D1 == D2
    close(D1)
    close(D2)
end

check_leaks()

@testset "test DArray reduce" begin
    D = DArray(id->fill(myid(), map(length,id)), (10,10), [MYID, OTHERIDS])

    @testset "test reduce" begin
        @test reduce(+, D) == ((50*MYID) + (50*OTHERIDS))
    end

    @testset "test map / reduce" begin
        D2 = map(x->1, D)
        @test D2 isa DArray
        @test reduce(+, D2) == 100
        close(D2)
    end

    @testset "test map! / reduce" begin
        map!(x->1, D, D)
        @test reduce(+, D) == 100
    end
    close(D)
end

check_leaks()

@testset "test rmul" begin
    A = randn(100,100)
    DA = distribute(A)
    @test rmul!(DA, 2) == rmul!(A, 2)
    close(DA)
end

check_leaks()

@testset "test rmul!(Diagonal, A)" begin
    A = randn(100, 100)
    b = randn(100)
    D = Diagonal(b)
    DA = distribute(A)
    @test lmul!(D, A) == lmul!(D, DA)
    close(DA)
    A = randn(100, 100)
    b = randn(100)
    DA = distribute(A)
    @test rmul!(A, D) == rmul!(DA, D)
    close(DA)
end

check_leaks()

@testset "test mapreduce on DArrays" begin
    for _ = 1:25, f = [x -> Int128(2x), x -> Int128(x^2), x -> Int128(x^2 + 2x - 1)], opt = [+, *]
        A = rand(1:5, rand(2:30))
        DA = distribute(A)
        @test DA isa DArray
        @test mapreduce(f, opt, DA) - mapreduce(f, opt, A) == 0
        close(DA)
    end
end

check_leaks()

@testset "test mapreducedim on DArrays" begin
    D = DArray(I->fill(myid(), map(length,I)), (73,73), [MYID, OTHERIDS])
    D2 = map(x->1, D)
    @test D2 isa DArray
    @test mapreduce(t -> t*t, +, D2, dims=1) == mapreduce(t -> t*t, +, convert(Array, D2), dims=1)
    @test mapreduce(t -> t*t, +, D2, dims=2) == mapreduce(t -> t*t, +, convert(Array, D2), dims=2)
    @test mapreduce(t -> t*t, +, D2, dims=(1,2)) == mapreduce(t -> t*t, +, convert(Array, D2), dims=(1,2))

    # Test non-regularly chunked DArrays
    r1 = DistributedArrays.remotecall(() -> sprandn(3, 10, 0.1), workers()[1])
    r2 = DistributedArrays.remotecall(() -> sprandn(7, 10, 0.1), workers()[2])
    D = DArray(reshape([r1; r2], (2,1)))
    @test Array(sum(D, dims=2)) == sum(Array(D), dims=2)

    # close(D)
    # close(D2)
    d_closeall()   # temp created by the mapreduce above
end

check_leaks()

@testset "test mapreducdim, reducedim on DArrays" begin
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    @testset "dimension $dms" for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
        @test mapreduce(t -> t*t, +, A, dims=dms) ≈ mapreduce(t -> t*t, +, DA, dims=dms)
        @test mapreduce(t -> t*t, +, A, dims=dms, init=1.0) ≈ mapreduce(t -> t*t, +, DA, dims=dms, init=1.0)
        @test reduce(*, A, dims=dms) ≈ reduce(*, DA, dims=dms)
        @test reduce(*, A, dims=dms, init=2.0) ≈ reduce(*, DA, dims=dms, init=2.0)
    end
    close(DA)
    d_closeall()   # temp created by the mapreduce above
end

check_leaks()

@testset "test statistical functions on DArrays" begin
    dims = (20,20,20)
    DA = drandn(dims)
    A = Array(DA)

    @testset "test $f for dimension $dms" for f in (mean, ), dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
        # std is pending implementation
        @test f(DA, dims=dms) ≈ f(A, dims=dms)
    end

    close(DA)
    d_closeall()   # temporaries created above
end

check_leaks()

@testset "test sum on DArrays" begin
    A = randn(100,100)
    DA = distribute(A)

    # sum either throws an ArgumentError or a CompositeException of ArgumentErrors
    try
        sum(DA, dims=-1)
    catch err
        if isa(err, CompositeException)
            @test !isempty(err.exceptions)
            for excep in err.exceptions
                # Unpack the remote exception
                orig_err = excep.ex.captured.ex
                @test isa(orig_err, ArgumentError)
            end
        else
            @test isa(err, ArgumentError)
        end
    end
    try
        sum(DA, dims=0)
    catch err
        if isa(err, CompositeException)
            @test !isempty(err.exceptions)
            for excep in err.exceptions
                # Unpack the remote exception
                orig_err = excep.ex.captured.ex
                @test isa(orig_err, ArgumentError)
            end
        else
            @test isa(err, ArgumentError)
        end
    end

    @test sum(DA) ≈ sum(A)
    @test sum(DA, dims=1) ≈ sum(A, dims=1)
    @test sum(DA, dims=2) ≈ sum(A, dims=2)
    @test sum(DA, dims=3) ≈ sum(A, dims=3)
    close(DA)
    d_closeall()   # temporaries created above
end

check_leaks()

@testset "test size on DArrays" begin

    A = randn(100,100)
    DA = distribute(A)

    @test_throws BoundsError size(DA, 0)
    @test size(DA,1) == size(A,1)
    @test size(DA,2) == size(A,2)
    @test size(DA,3) == size(A,3)
    close(DA)
end

check_leaks()

# test length / lastindex
@testset "test collections API" begin
    A = randn(23,23)
    DA = distribute(A)

    @testset "test length" begin
        @test length(DA) == length(A)
    end

    @testset "test lastindex" begin
        @test lastindex(DA) == lastindex(A)
    end
    close(DA)
end

check_leaks()

@testset "test max / min / sum" begin
    a = map(x -> Int(round(rand() * 100)) - 50, Array{Int}(undef,100,1000))
    d = distribute(a)

    @test sum(d)          == sum(a)
    @test maximum(d)      == maximum(a)
    @test minimum(d)      == minimum(a)
    @test maximum(abs, d) == maximum(abs, a)
    @test minimum(abs, d) == minimum(abs, a)
    @test sum(abs, d)     == sum(abs, a)
    @test sum(abs2, d)    == sum(abs2, a)
    @test extrema(d)      == extrema(a)
    close(d)
end

check_leaks()

@testset "test all / any" begin
    a = map(x->Int(round(rand() * 100)) - 50, Array{Int}(undef,100,1000))
    a = [true for i in 1:100]
    d = distribute(a)

    @test all(d)
    @test any(d)

    close(d)

    a[50] = false
    d = distribute(a)
    @test !all(d)
    @test any(d)

    close(d)

    a = [false for i in 1:100]
    d = distribute(a)
    @test !all(d)
    @test !any(d)

    close(d)

    d = dones(10,10)
    @test !all(x-> x>1.0, d)
    @test all(x-> x>0.0, d)

    close(d)

    a = ones(10,10)
    a[10] = 2.0
    d = distribute(a)
    @test any(x-> x == 1.0, d)
    @test any(x-> x == 2.0, d)
    @test !any(x-> x == 3.0, d)

    close(d)
end

check_leaks()

@testset "test count"  begin
    a = ones(10,10)
    a[10] = 2.0
    d = distribute(a)

    @test count(x-> x == 2.0, d) == 1
    @test count(x-> x == 1.0, d) == 99
    @test count(x-> x == 0.0, d) == 0

    close(d)
end

check_leaks()

@testset "test prod" begin
    a = fill(2, 10);
    d = distribute(a);
    @test prod(d) == 2^10

    close(d)
end

check_leaks()

@testset "test zeros" begin
    @testset "1D dzeros default element type" begin
        A = dzeros(10)
        @test A == zeros(10)
        @test eltype(A) == Float64
        @test size(A) == (10,)
        close(A)
    end

    @testset "1D dzeros with specified element type" begin
        A = dzeros(Int, 10)
        @test A == zeros(10)
        @test eltype(A) == Int
        @test size(A) == (10,)
        close(A)
    end

    @testset "2D dzeros default element type, Dims constuctor" begin
        A = dzeros((10,10))
        @test A == zeros((10,10))
        @test eltype(A) == Float64
        @test size(A) == (10,10)
        close(A)
    end

    @testset "2D dzeros specified element type, Dims constructor" begin
        A = dzeros(Int, (10,10))
        @test A == zeros(Int, (10,10))
        @test eltype(A) == Int
        @test size(A) == (10,10)
        close(A)
    end

    @testset "2D dzeros, default element type" begin
        A = dzeros(10,10)
        @test A == zeros(10,10)
        @test eltype(A) == Float64
        @test size(A) == (10,10)
        close(A)
    end

    @testset "2D dzeros, specified element type" begin
        A = dzeros(Int, 10, 10)
        @test A == zeros(Int, 10, 10)
        @test eltype(A) == Int
        @test size(A) == (10,10)
        close(A)
    end
end

check_leaks()

@testset "test dones" begin
    @testset "1D dones default element type" begin
        A = dones(10)
        @test A == ones(10)
        @test eltype(A) == Float64
        @test size(A) == (10,)
        close(A)
    end

    @testset "1D dones with specified element type" begin
        A = dones(Int, 10)
        @test eltype(A) == Int
        @test size(A) == (10,)
        close(A)
    end

    @testset "2D dones default element type, Dims constuctor" begin
        A = dones((10,10))
        @test A == ones((10,10))
        @test eltype(A) == Float64
        @test size(A) == (10,10)
        close(A)
    end

    @testset "2D dones specified element type, Dims constructor" begin
        A = dones(Int, (10,10))
        @test A == ones(Int, (10,10))
        @test eltype(A) == Int
        @test size(A) == (10,10)
        close(A)
    end

    @testset "2D dones, default element type" begin
        A = dones(10,10)
        @test A == ones(10,10)
        @test eltype(A) == Float64
        @test size(A) == (10,10)
        close(A)
    end

    @testset "2D dones, specified element type" begin
        A = dones(Int, 10, 10)
        @test A == ones(Int, 10, 10)
        @test eltype(A) == Int
        @test size(A) == (10,10)
        close(A)
    end
end

check_leaks()

@testset "test drand" begin
    @testset "1D drand" begin
        A = drand(100)
        @test eltype(A) == Float64
        @test size(A) == (100,)
        @test all(x-> x >= 0.0 && x <= 1.0, A)
        close(A)
    end

    @testset "1D drand, specified element type" begin
        A = drand(Int, 100)
        @test eltype(A) == Int
        @test size(A) == (100,)
        close(A)
    end

    @testset "1D drand, UnitRange" begin
        A = drand(1:10, 100)
        @test eltype(A) == Int
        @test size(A) == (100,)
        close(A)
    end

    @testset "1D drand, Array" begin
        A = drand([-1,0,1], 100)
        @test eltype(A) == Int
        @test size(A) == (100,)
        close(A)
    end

    @testset "2D drand, Dims constructor" begin
        A = drand((50,50))
        @test eltype(A) == Float64
        @test size(A) == (50,50)
        @test all(x-> x >= 0.0 && x <= 1.0, A)
        close(A)
    end

    @testset "2D drand" begin
        A = drand(100,100)
        @test eltype(A) == Float64
        @test size(A) == (100,100)
        @test all(x-> x >= 0.0 && x <= 1.0, A)
        close(A)
    end

    @testset "2D drand, Dims constructor, specified element type" begin
        A = drand(Int, (100,100))
        @test eltype(A) == Int
        @test size(A) == (100,100)
        close(A)
    end

    @testset "2D drand, specified element type" begin
        A = drand(Int, 100, 100)
        @test eltype(A) == Int
        @test size(A) == (100,100)
        close(A)
    end
end

check_leaks()

@testset "test randn" begin
    @testset "1D drandn" begin
        A = drandn(100)
        @test eltype(A) == Float64
        @test size(A) == (100,)
        close(A)
    end

    @testset "2D drandn, Dims constructor" begin
        A = drandn((50,50))
        @test eltype(A) == Float64
        @test size(A) == (50,50)
        close(A)
    end

    @testset "2D drandn" begin
        A = drandn(100,100)
        @test eltype(A) == Float64
        @test size(A) == (100,100)
        close(A)
    end
end

check_leaks()

@testset "test transpose/adjoint" begin
    @testset "test transpose real" begin
        A = drand(Float64, 100, 200)
        @test copy(transpose(A)) == transpose(Array(A))
        close(A)
    end
    @testset "test transpose complex" begin
        A = drand(ComplexF64, 200, 100)
        @test copy(transpose(A)) == transpose(Array(A))
        close(A)
    end
    @testset "test adjoint real" begin
        A = drand(Float64, 200, 100)
        @test copy(adjoint(A)) == adjoint(Array(A))
        close(A)
    end
    @testset "test adjoint complex" begin
        A = drand(ComplexF64, 100, 200)
        @test copy(adjoint(A)) == adjoint(Array(A))
        close(A)
    end

    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "makelocal" begin
    A = randn(5*nprocs(), 5*nprocs())
    dA = distribute(A, procs=procs())
    for i in 1:size(dA, 2)
        a = DistributedArrays.makelocal(dA, :, i)
        @test all(Array(view(dA, :, i)) .== a)
        @test all(      view( A, :, i) .== a)
    end
    for i in 1:size(dA, 1)
        a = DistributedArrays.makelocal(dA, i, :)
        @test all(Array(view(dA, i:i, :)) .== a)
        @test all(      view( A, i:i, :) .== a)
    end
    a = DistributedArrays.makelocal(dA, 1:5, 1:5)
    @test all(Array(view(dA, 1:5, 1:5)) .== a)
    @test all(      view( A, 1:5, 1:5) .== a)
    close(dA)
end

@testset "test convert from subdarray" begin
    a = drand(20, 20);

    s = view(a, 1:5, 5:8)
    @test isa(s, SubDArray)
    @test s == DArray(s)

    s = view(a, 6:5, 5:8)
    @test isa(s, SubDArray)
    @test s == DArray(s)
    close(a)
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test scalar math" begin
    a = drand(20, 20);
    b = convert(Array, a)
    @testset "$f" for f in (-, abs, abs2, acos, acosd, acot,
              acotd, acsch, angle, asech, asin,
              asind, asinh, atan, atand, atanh,
              big, cbrt, ceil, cis, complex, conj,
              cos, cosc, cosd, cosh, cospi, cot,
              cotd, coth, csc, cscd, csch, dawson,
              deg2rad, digamma, erf, erfc, erfcinv,
              erfcx, erfi, erfinv, exp, exp10, exp2,
              expm1, exponent, float, floor, gamma, imag,
              invdigamma, isfinite, isinf, isnan,
              lgamma, log, log10, log1p, log2, rad2deg, real,
              sec, secd, sech, sign, sin, sinc, sind,
              sinh, sinpi, sqrt, tan, tand, tanh, trigamma)
        @test f.(a) == f.(b)
    end
    a = a .+ 1
    b = b .+ 1
    @testset "$f" for f in (asec, asecd, acosh, acsc, acscd, acoth)
        @test f.(a) == f.(b)
    end
    close(a)
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test mapslices" begin
    A = randn(5,5,5)
    D = distribute(A, procs = workers(), dist = [1, 1, min(nworkers(), 5)])
    @test mapslices(svdvals, D, dims=(1,2)) ≈ mapslices(svdvals, A, dims=(1,2))
    @test mapslices(svdvals, D, dims=(1,3)) ≈ mapslices(svdvals, A, dims=(1,3))
    @test mapslices(svdvals, D, dims=(2,3)) ≈ mapslices(svdvals, A, dims=(2,3))
    @test mapslices(sort, D, dims=(1,)) ≈ mapslices(sort, A, dims=(1,))
    @test mapslices(sort, D, dims=(2,)) ≈ mapslices(sort, A, dims=(2,))
    @test mapslices(sort, D, dims=(3,)) ≈ mapslices(sort, A, dims=(3,))

    # issue #3613
    B = mapslices(sum, dones(Float64, (2,3,4), workers(), [1,1,min(nworkers(),4)]), dims=[1,2])
    @test size(B) == (1,1,4)
    @test all(B.==6)

    # issue #5141
    C1 = mapslices(x-> maximum(-x), D, dims=[])
    @test C1 == -D

    # issue #5177
    c = dones(Float64, (2,3,4,5), workers(), [1,1,1,min(nworkers(),5)])
    m1 = mapslices(x-> ones(2,3), c, dims=[1,2])
    m2 = mapslices(x-> ones(2,4), c, dims=[1,3])
    m3 = mapslices(x-> ones(3,4), c, dims=[2,3])
    @test size(m1) == size(m2) == size(m3) == size(c)

    n1 = mapslices(x-> ones(6), c, dims=[1,2])
    n2 = mapslices(x-> ones(6), c, dims=[1,3])
    n3 = mapslices(x-> ones(6), c, dims=[2,3])
    n1a = mapslices(x-> ones(1,6), c, dims=[1,2])
    n2a = mapslices(x-> ones(1,6), c, dims=[1,3])
    n3a = mapslices(x-> ones(1,6), c, dims=[2,3])
    @test (size(n1a) == (1,6,4,5) && size(n2a) == (1,3,6,5) && size(n3a) == (2,1,6,5))
    @test (size(n1) == (6,1,4,5) && size(n2) == (6,3,1,5) && size(n3) == (2,6,1,5))
    close(D)
    close(c)
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test scalar ops" begin
    a = drand(20,20)
    b = convert(Array, a)
    c = drand(20,20)
    d = convert(Array, c)

    @testset "$f" for f in (:+, :-, :*, :/, :%)
        x = rand()
        @test @eval ($f).($a, $x) == ($f).($b, $x)
        @test @eval ($f).($x, $a) == ($f).($x, $b)
        @test @eval ($f).($a, $c) == ($f).($b, $d)
    end

    close(a)
    close(c)

    a = dones(Int, 20, 20)
    b = convert(Array, a)
    @testset "$f" for f in (:<<, :>>)
        @test @eval ($f).($a, 2)  == ($f).($b, 2)
        @test @eval ($f).(2, $a)  == ($f).(2, $b)
        @test @eval ($f).($a, $a) == ($f).($b, $b)
    end

    @testset "$f" for f in (:rem,)
        x = rand()
        @test @eval ($f).($a, $x) == ($f).($b, $x)
    end
    close(a)
    close(c)
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test broadcast ops" begin
    wrkrs = workers()
    nwrkrs = length(wrkrs)
    nrows = 20 * nwrkrs
    ncols = 10 * nwrkrs
    a = drand((nrows,ncols), wrkrs, (1, nwrkrs))
    m = mean(a, dims=1)
    c = a .- m
    d = convert(Array, a) .- convert(Array, m)
    @test c == d
    e = @DArray [ones(10) for i=1:4]
    f = 2 .* e
    @test Array(f) == 2 .* Array(e)
    @test Array(map(x -> sum(x) .+ 2, e)) == map(x -> sum(x) .+ 2, e)

    @testset "test nested broadcast" begin
       g = a .- m .* sin.(c)
       @test Array(g) == Array(a) .- Array(m) .* sin.(Array(c))
    end

    # @testset "lazy wrapped broadcast" begin
    #    l = similar(a)
    #    l[1:10, :] .= view(a, 1:10, : )
    # end
    d_closeall()
end

check_leaks()

@testset "test matrix multiplication" begin
    A = drandn(20,20)
    b = drandn(20)
    B = drandn(20,20)

    @test norm(convert(Array, A*b) - convert(Array, A)*convert(Array, b), Inf) < sqrt(eps())
    @test norm(convert(Array, A*B) - convert(Array, A)*convert(Array, B), Inf) < sqrt(eps())
    @test norm(convert(Array, A'*b) - convert(Array, A)'*convert(Array, b), Inf) < sqrt(eps())
    @test norm(convert(Array, A'*B) - convert(Array, A)'*convert(Array, B), Inf) < sqrt(eps())
    close(A)
    close(b)
    close(B)
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "dot product" begin
    A = drandn(20,20)
    b = drandn(20)
    c = A * b

    @test dot(c, b) ≈ dot(convert(Array, c), convert(Array, b))
    close(A)
    close(b)
    close(c)
end

check_leaks()

@testset "test norm" begin
    x = drandn(20)

    @test abs(norm(x) - norm(convert(Array, x))) < sqrt(eps())
    @test abs(norm(x, 1) - norm(convert(Array, x), 1)) < sqrt(eps())
    @test abs(norm(x, 2) - norm(convert(Array, x), 2)) < sqrt(eps())
    @test abs(norm(x, Inf) - norm(convert(Array, x), Inf)) < sqrt(eps())
    close(x)
end

check_leaks()

@testset "test axpy!" begin
    for (x, y) in ((drandn(20), drandn(20)),
                   (drandn(20, 2), drandn(20, 2)))

        @test Array(axpy!(2.0, x, copy(y))) ≈ axpy!(2.0, Array(x), Array(y))
        @test_throws DimensionMismatch axpy!(2.0, x, zeros(length(x) + 1))
        close(x)
        close(y)
    end

    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test ppeval" begin
    A = drandn((10, 10, nworkers()), workers(), [1, 1, nworkers()])
    B = drandn((10, nworkers()), workers(), [1, nworkers()])

    R = zeros(10, nworkers())
    for i = 1:nworkers()
        R[:, i] = convert(Array, A)[:, :, i]*convert(Array, B)[:, i]
    end
    @test convert(Array, ppeval(*, A, B)) ≈ R
    @test sum(ppeval(eigvals, A)) ≈ sum(ppeval(eigvals, A, Matrix{Float64}(I,10,10)))
    close(A)
    close(B)
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test nnz" begin
    A = sprandn(10, 10, 0.5)
    @test nnz(distribute(A)) == nnz(A)
end

@testset "test matmatmul" begin
    A = drandn(30, 30)
    B = drandn(30, 20)
    a = convert(Array, A)
    b = convert(Array, B)

    AB = A * B
    AtB = transpose(A) * B
    AcB = A' * B

    ab = a * b
    atb = transpose(a) * b
    acb = a' * b

    @test AB ≈ ab
    @test AtB ≈ atb
    @test AcB ≈ acb
    d_closeall()  # close the temporaries created above
end

@testset "sort, T = $T, 10^$i elements" for i in 0:6, T in [Int, Float64]
    d = DistributedArrays.drand(T, 10^i)
    @testset "sample = $sample" for sample in Any[true, false, (minimum(d),maximum(d)), rand(T, 10^i>512 ? 512 : 10^i)]
        d2 = DistributedArrays.sort(d; sample=sample)
        a  = convert(Array, d)
        a2 = convert(Array, d2)
        @test length(d) == length(d2)
        @test sort(a) == a2
    end
    d_closeall()  # close the temporaries created above
end

check_leaks()

@testset "ddata" begin
    d = ddata(;T=Int, init=I->myid())
    for p in workers()
        @test p == remotecall_fetch(d->d[:L], p, d)
    end
    @test Int[workers()...] == gather(d)

    close(d)

    d = ddata(;T=Int, data=workers())
    for p in workers()
        @test p == remotecall_fetch(d->d[:L], p, d)
    end
    @test Int[workers()...] == gather(d)

    close(d)

    d = ddata(;T=Any, init=I->"Hello World!")
    for p in workers()
        @test "Hello World!" == remotecall_fetch(d->d[:L], p, d)
    end
    Any["Hello World!" for p in workers()] == gather(d)


    close(d)
end

@testset "rand!" begin
    d = dzeros(30, 30)
    rand!(d)

    close(d)
end

check_leaks()

d_closeall()

@testset "test for any leaks" begin
    sleep(1.0)     # allow time for any cleanup to complete
    allrefszero = Bool[remotecall_fetch(()->length(DistributedArrays.refs) == 0, p) for p in procs()]
    @test all(allrefszero)

    allregistrieszero = Bool[remotecall_fetch(()->length(DistributedArrays.registry) == 0, p) for p in procs()]
    @test all(allregistrieszero)
end

@testset "internal API" begin
    @testset "arraykind" begin
       @test DistributedArrays.arraykind(Array{Float32, 2}) == Array
       @test DistributedArrays.arraykind(AbstractArray{Float32, 2}) == Array
       @test DistributedArrays.arraykind(typeof(1:10)) == UnitRange
    end
end

