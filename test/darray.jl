const MYID = myid()
const OTHERIDS = filter(id-> id != MYID, procs())[rand(1:(nprocs()-1))]

function check_leaks()
    if length(DistributedArrays.refs) > 0
        sleep(0.1)  # allow time for any cleanup to complete and test again
        length(DistributedArrays.refs) > 0 && warn("Probable leak of ", length(DistributedArrays.refs), " darrays")
    end
end

@testset "test distribute" begin
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
end

check_leaks()

@testset "test DArray equality" begin
    D = drand((200,200), [MYID, OTHERIDS])
    DC = copy(D)

    @testset "test isequal(::DArray, ::DArray)" begin
        @test D == DC
    end

    @testset "test copy(::DArray) does a copy of each localpart" begin
        @spawnat OTHERIDS localpart(DC)[1] = 0
        @test fetch(@spawnat OTHERIDS localpart(D)[1] != 0)
    end

    close(D)
    close(DC)
end

check_leaks()

@testset "test @DArray comprehension constructor" begin

    @testset "test valid use of @DArray" begin
        D = @DArray [i+j for i=1:10, j=1:10]
        @test D == [i+j for i=1:10, j=1:10]
        close(D)
    end

    @testset "test invalid use of @DArray" begin
        @test_throws ArgumentError eval(:((@DArray [1,2,3,4])))
    end
end

check_leaks()

@testset "test DArray / Array conversion" begin
    D = drand((200,200), [MYID, OTHERIDS])

    @testset "test convert(::Array, ::(Sub)DArray)" begin
        S = convert(Matrix{Float64}, D[1:150, 1:150])
        A = convert(Matrix{Float64}, D)

        @test A[1:150,1:150] == S
        D2 = convert(DArray{Float64,2,Matrix{Float64}}, A)
        @test D2 == D
        @test fetch(@spawnat MYID localpart(D)[1,1]) == D[1,1]
        @test fetch(@spawnat OTHERIDS localpart(D)[1,1]) == D[1,101]
        close(D2)

        S2 = convert(Vector{Float64}, D[4, 23:176])
        @fact A[4, 23:176] --> S2

        S3 = convert(Vector{Float64}, D[23:176, 197])
        @fact A[23:176, 197] --> S3

        S4 = zeros(4)
        setindex!(S4, D[3:4, 99:100], :)
        @fact S4 --> vec(D[3:4, 99:100])
        @fact S4 --> vec(A[3:4, 99:100])
        
        S5 = zeros(2,2)
        setindex!(S5, D[1,1:4], :, 1:2)
        @fact vec(S5) --> D[1, 1:4]
        @fact vec(S5) --> A[1, 1:4]
    end
    close(D)
end

check_leaks()

@testset "test DArray reduce" begin
    D = DArray(id->fill(myid(), map(length,id)), (10,10), [MYID, OTHERIDS])

    @testset "test reduce" begin
        @test reduce(+, D) == ((50*MYID) + (50*OTHERIDS))
    end

    @testset "test map / reduce" begin
        D2 = map(x->1, D)
        @test reduce(+, D2) == 100
        close(D2)
    end

    @testset "test map! / reduce" begin
        map!(x->1, D)
        @test reduce(+, D) == 100
    end
    close(D)
end

check_leaks()

@testset "test scale" begin
    A = randn(100,100)
    DA = distribute(A)
    @test scale!(DA, 2) == scale!(A, 2)
    close(DA)
end

check_leaks()

@testset "test mapreduce on DArrays" begin
    for _ = 1:25, f = [x -> Int128(2x), x -> Int128(x^2), x -> Int128(x^2 + 2x - 1)], opt = [+, *]
        A = rand(1:5, rand(2:30))
        DA = distribute(A)
        @test mapreduce(f, opt, DA) - mapreduce(f, opt, A) == 0
        close(DA)
    end
end

check_leaks()

@testset "test mapreducedim on DArrays" begin
    D = DArray(I->fill(myid(), map(length,I)), (73,73), [MYID, OTHERIDS])
    D2 = map(x->1, D)
    @test mapreducedim(t -> t*t, +, D2, 1) == mapreducedim(t -> t*t, +, convert(Array, D2), 1)
    @test mapreducedim(t -> t*t, +, D2, 2) == mapreducedim(t -> t*t, +, convert(Array, D2), 2)
    @test mapreducedim(t -> t*t, +, D2, (1,2)) == mapreducedim(t -> t*t, +, convert(Array, D2), (1,2))

    # Test non-regularly chunked DArrays
    r1 = DistributedArrays.remotecall(() -> sprandn(3, 10, 0.1), workers()[1])
    r2 = DistributedArrays.remotecall(() -> sprandn(7, 10, 0.1), workers()[2])
    D = DArray(reshape([r1; r2], (2,1)))
    @test Array(sum(D, 2)) == sum(Array(D), 2)

    # close(D)
    # close(D2)
    darray_closeall()   # temp created by the mapreduce above
end

check_leaks()

@testset "test mapreducdim, reducedim on DArrays" begin
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    @testset "dimension $dms" for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
        @test mapreducedim(t -> t*t, +, A, dms) ≈ mapreducedim(t -> t*t, +, DA, dms)
        @test mapreducedim(t -> t*t, +, A, dms, 1.0) ≈ mapreducedim(t -> t*t, +, DA, dms, 1.0)
        @test reducedim(*, A, dms) ≈ reducedim(*, DA, dms)
        @test reducedim(*, A, dms, 2.0) ≈ reducedim(*, DA, dms, 2.0)
    end
    close(DA)
    darray_closeall()   # temp created by the mapreduce above
end

check_leaks()

@testset "test statistical functions on DArrays" begin
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    @testset "test $f for dimension $dms" for f in (mean, ), dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
        # std is pending implementation
        @test f(DA,dms) ≈ f(A,dms)
    end

    close(DA)
    darray_closeall()   # temporaries created above
end

check_leaks()

@testset "test sum on DArrays" begin
    A = randn(100,100)
    DA = distribute(A)

    # sum either throws an ArgumentError or a CompositeException of ArgumentErrors
    try
        sum(DA, -1)
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
        sum(DA, 0)
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
    @test sum(DA,1) ≈ sum(A,1)
    @test sum(DA,2) ≈ sum(A,2)
    @test sum(DA,3) ≈ sum(A,3)
    close(DA)
    darray_closeall()   # temporaries created above
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

# test length / endof
@testset "test collections API" begin
    A = randn(23,23)
    DA = distribute(A)

    @testset "test length" begin
        @test length(DA) == length(A)
    end

    @testset "test endof" begin
        @test endof(DA) == endof(A)
    end
    close(DA)
end

check_leaks()

@testset "test max / min / sum" begin
    a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
    d = distribute(a)

    @test sum(d) == sum(a)
    @test maximum(d) == maximum(a)
    @test minimum(d) == minimum(a)
    @test maxabs(d) == maxabs(a)
    @test minabs(d) == minabs(a)
    @test sumabs(d) == sumabs(a)
    @test sumabs2(d) == sumabs2(a)
    close(d)
end

check_leaks()

@testset "test all / any" begin
    a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
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

@testset "test c/transpose" begin
    @testset "test ctranspose real" begin
        A = drand(Float64, 100, 200)
        @test A' == Array(A)'
        close(A)
    end
    @testset "test ctranspose complex" begin
        A = drand(Complex128, 200, 100)
        @test A' == Array(A)'
        close(A)
    end
    @testset "test transpose real" begin
        A = drand(Float64, 200, 100)
        @test A.' == Array(A).'
        close(A)
    end
    @testset "test ctranspose complex" begin
        A = drand(Complex128, 100, 200)
        @test A.' == Array(A).'
        close(A)
    end

    darray_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test convert from subdarray" begin
    a = drand(20, 20);

    s = view(a, 1:5, 5:8)
    @test isa(s, SubDArray)
    @test s == convert(DArray, s)

    s = view(a, 6:5, 5:8)
    @test isa(s, SubDArray)
    @test s == convert(DArray, s)
    close(a)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test scalar math" begin
    a = drand(20, 20);
    b = convert(Array, a)
    @testset "$f" for f in (:abs, :abs2, :acos, :acosd, :acot,
              :acotd, :acsch, :angle, :asech, :asin,
              :asind, :asinh, :atan, :atand, :atanh,
              :big, :cbrt, :ceil, :cis, :complex, :conj,
              :cos, :cosc, :cosd, :cosh, :cospi, :cot,
              :cotd, :coth, :csc, :cscd, :csch, :dawson,
              :deg2rad, :digamma, :erf, :erfc, :erfcinv,
              :erfcx, :erfi, :erfinv, :exp, :exp10, :exp2,
              :expm1, :exponent, :float, :floor, :gamma, :imag,
              :invdigamma, :isfinite, :isinf, :isnan, :lfact,
              :lgamma, :log, :log10, :log1p, :log2, :rad2deg, :real,
              :sec, :secd, :sech, :sign, :sin, :sinc, :sind,
              :sinh, :sinpi, :sqrt, :tan, :tand, :tanh, :trigamma)
        @test (eval(f))(a) == (eval(f))(b)
    end
    a = a + 1
    b = b + 1
    @testset "$f" for f in (:asec, :asecd, :acosh, :acsc, :acscd, :acoth)
        @test (eval(f))(a) == (eval(f))(b)
    end
    close(a)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

# The mapslices tests have been taken from Base.
# Commented out tests that need to be enabled in due course when DArray support is more complete
@testset "test mapslices" begin
    a = drand((5,5), workers(), [1, min(nworkers(), 5)])
    if VERSION < v"0.5.0-dev+4361"
        h = mapslices(v -> hist(v,0:0.1:1)[2], a, 1)
    else
        h = mapslices(v -> fit(Histogram,v,0:0.1:1).weights, a, 1)
    end
#    H = mapslices(v -> hist(v,0:0.1:1)[2], a, 2)
#    s = mapslices(sort, a, [1])
#    S = mapslices(sort, a, [2])
    for i = 1:5
        if VERSION < v"0.5.0-dev+4361"
            @test h[:,i] == hist(a[:,i],0:0.1:1)[2]
        else
            @test h[:,i] == fit(Histogram, a[:,i],0:0.1:1).weights
        end
#        @test vec(H[i,:]) => hist(vec(a[i,:]),0:0.1:1)[2]
#        @test s[:,i] => sort(a[:,i])
#        @test vec(S[i,:]) => sort(vec(a[i,:]))
    end

    # issue #3613
    b = mapslices(sum, dones(Float64, (2,3,4), workers(), [1,1,min(nworkers(),4)]), [1,2])
    @test size(b) == (1,1,4)
    @test all(b.==6)

    # issue #5141
    ## Update Removed the version that removes the dimensions when dims==1:ndims(A)
    c1 = mapslices(x-> maximum(-x), a, [])
#    @test c1 => -a

    # issue #5177
    c = dones(Float64, (2,3,4,5), workers(), [1,1,1,min(nworkers(),5)])
    m1 = mapslices(x-> ones(2,3), c, [1,2])
    m2 = mapslices(x-> ones(2,4), c, [1,3])
    m3 = mapslices(x-> ones(3,4), c, [2,3])
    @test size(m1) == size(m2) == size(m3) == size(c)

    n1 = mapslices(x-> ones(6), c, [1,2])
    n2 = mapslices(x-> ones(6), c, [1,3])
    n3 = mapslices(x-> ones(6), c, [2,3])
    n1a = mapslices(x-> ones(1,6), c, [1,2])
    n2a = mapslices(x-> ones(1,6), c, [1,3])
    n3a = mapslices(x-> ones(1,6), c, [2,3])
    @test (size(n1a) == (1,6,4,5) && size(n2a) == (1,3,6,5) && size(n3a) == (2,1,6,5))
    @test (size(n1) == (6,1,4,5) && size(n2) == (6,3,1,5) && size(n3) == (2,6,1,5))
    close(a)
    close(c)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test scalar ops" begin
    a = drand(20,20)
    b = convert(Array, a)
    c = drand(20,20)
    d = convert(Array, c)

    @testset "$f" for f in (:+, :-, :.+, :.-, :.*, :./, :.%, :div, :mod)
        x = rand()
        @test (eval(f))(a, x) == (eval(f))(b, x)
        @test (eval(f))(x, a) == (eval(f))(x, b)
        @test (eval(f))(a, c) == (eval(f))(b, d)
    end

    close(a)
    close(c)

    a = dones(Int, 20, 20)
    b = convert(Array, a)
    @testset "$f" for f in (:.<<, :.>>)
        @test (eval(f))(a, 2) == (eval(f))(b, 2)
        @test (eval(f))(2, a) == (eval(f))(2, b)
        @test (eval(f))(a, a) == (eval(f))(b, b)
    end

    @testset "$f" for f in (:rem,)
        x = rand()
        @test (eval(f))(a, x) == (eval(f))(b, x)
    end
    close(a)
    close(c)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

@testset "test broadcast ops" begin
    wrkrs = workers()
    nwrkrs = length(wrkrs)
    nrows = 20 * nwrkrs
    ncols = 10 * nwrkrs
    a = drand((nrows,ncols), wrkrs, (1, nwrkrs))
    m = mean(a, 1)
    c = a .- m
    d = convert(Array, a) .- convert(Array, m)
    @test c == d
    darray_closeall()
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
    darray_closeall()  # close the temporaries created above
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
    x = drandn(20)
    y = drandn(20)

    @test norm(convert(Array, LinAlg.axpy!(2.0, x, copy(y))) - LinAlg.axpy!(2.0, convert(Array, x), convert(Array, y))) < sqrt(eps())
    @test_throws DimensionMismatch LinAlg.axpy!(2.0, x, zeros(length(x) + 1))
    close(x)
    close(y)
    darray_closeall()  # close the temporaries created above
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
    @test sum(ppeval(eigvals, A)) ≈ sum(ppeval(eigvals, A, eye(10, 10)))
    close(A)
    close(B)
    darray_closeall()  # close the temporaries created above
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
    AtB = A.' * B
    AcB = A' * B

    ab = a * b
    atb = a.' * b
    acb = a' * b

    @test AB ≈ ab
    @test AtB ≈ atb
    @test AcB ≈ acb
    darray_closeall()  # close the temporaries created above
end

@testset "sort, T = $T" for i in 0:6, T in [Int, Float64]
    d = DistributedArrays.drand(T, 10^i)
    @testset "sample = $sample" for sample in Any[true, false, (minimum(d),maximum(d)), rand(T, 10^i>512 ? 512 : 10^i)]
        d2 = DistributedArrays.sort(d; sample=sample)

        @test length(d) == length(d2)
        @test sort(convert(Array, d)) == convert(Array, d2)
    end
    darray_closeall()  # close the temporaries created above
end

check_leaks()

darray_closeall()

@testset "test for any leaks" begin
    sleep(1.0)     # allow time for any cleanup to complete
    allrefszero = Bool[remotecall_fetch(()->length(DistributedArrays.refs) == 0, p) for p in procs()]
    @test all(allrefszero)

    allregistrieszero = Bool[remotecall_fetch(()->length(DistributedArrays.registry) == 0, p) for p in procs()]
    @test all(allregistrieszero)
end
