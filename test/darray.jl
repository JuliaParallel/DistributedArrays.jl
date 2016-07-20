const MYID = myid()
const OTHERIDS = filter(id-> id != MYID, procs())[rand(1:(nprocs()-1))]

function check_leaks()
    if length(DistributedArrays.refs) > 0
        sleep(0.1)  # allow time for any cleanup to complete and test again
        length(DistributedArrays.refs) > 0 && warn("Probable leak of ", length(DistributedArrays.refs), " darrays")
    end
end

facts("test distribute") do
    A = randn(100,100)

    context("test default distribute") do
        DA = distribute(A)
        @fact length(procs(DA)) --> nworkers()
        close(DA)
    end

    context("test distribute with procs arguments") do
        DA = distribute(A, procs = [1, 2])
        @fact length(procs(DA)) --> 2
        close(DA)
    end

    context("test distribute with procs and dist arguments") do
        DA = distribute(A, procs = [1, 2], dist = [1,2])
        @fact size(procs(DA)) --> (1,2)
        close(DA)
    end
end

check_leaks()

facts("test DArray equality") do
    D = drand((200,200), [MYID, OTHERIDS])
    DC = copy(D)

    context("test isequal(::Array, ::DArray)") do
        @fact D == DC --> true
    end

    context("test copy(::DArray) does a copy of each localpart") do
        @spawnat OTHERIDS localpart(DC)[1] = 0
        @fact fetch(@spawnat OTHERIDS localpart(D)[1] != 0) --> true
    end

    close(D)
    close(DC)
end

check_leaks()

facts("test @DArray comprehension constructor") do

    context("test valid use of @DArray") do
        D = @DArray [i+j for i=1:10, j=1:10]
        @fact D --> [i+j for i=1:10, j=1:10]
        close(D)
    end

    context("test invalid use of @DArray") do
        @fact_throws ArgumentError eval(:((@DArray [1,2,3,4])))
    end
end

check_leaks()

facts("test DArray / Array conversion") do
    D = drand((200,200), [MYID, OTHERIDS])

    context("test convert(::Array, ::(Sub)DArray)") do
        S = convert(Matrix{Float64}, D[1:150, 1:150])
        A = convert(Matrix{Float64}, D)

        @fact A[1:150,1:150] --> S
        D2 = convert(DArray{Float64,2,Matrix{Float64}}, A)
        @fact D2 --> D
        @fact fetch(@spawnat MYID localpart(D)[1,1]) --> D[1,1]
        @fact fetch(@spawnat OTHERIDS localpart(D)[1,1]) --> D[1,101]
        close(D2)
    end
    close(D)
end

check_leaks()

facts("test DArray reduce") do
    D = DArray(id->fill(myid(), map(length,id)), (10,10), [MYID, OTHERIDS])

    context("test reduce") do
        @fact reduce(+, D) --> ((50*MYID) + (50*OTHERIDS))
    end

    context("test map / reduce") do
        D2 = map(x->1, D)
        @fact reduce(+, D2) --> 100
        close(D2)
    end

    context("test map! / reduce") do
        map!(x->1, D)
        @fact reduce(+, D) --> 100
    end
    close(D)
end

check_leaks()

facts("test scale") do
    A = randn(100,100)
    DA = distribute(A)
    @fact scale!(DA, 2) --> scale!(A, 2)
    close(DA)
end

check_leaks()

facts("test mapreduce on DArrays") do
    # temporaroly reduce to three iterations because of https://github.com/JuliaLang/julia/issues/15766
    # for _ = 1:25, f = [x -> Int128(2x), x -> Int128(x^2), x -> Int128(x^2 + 2x - 1)], opt = [+, *]
    for _ = 1:3, f = [x -> Int128(2x), x -> Int128(x^2), x -> Int128(x^2 + 2x - 1)], opt = [+, *]
        A = rand(1:5, rand(2:30))
        DA = distribute(A)
        @fact mapreduce(f, opt, DA) - mapreduce(f, opt, A) == 0 --> true
        close(DA)
    end
end

check_leaks()

facts("test mapreducedim on DArrays") do
    D = DArray(I->fill(myid(), map(length,I)), (73,73), [MYID, OTHERIDS])
    D2 = map(x->1, D)
    @fact mapreducedim(t -> t*t, +, D2, 1) --> mapreducedim(t -> t*t, +, convert(Array, D2), 1)
    @fact mapreducedim(t -> t*t, +, D2, 2) --> mapreducedim(t -> t*t, +, convert(Array, D2), 2)
    @fact mapreducedim(t -> t*t, +, D2, (1,2)) --> mapreducedim(t -> t*t, +, convert(Array, D2), (1,2))

    close(D)
    close(D2)
    darray_closeall()   # temp created by the mapreduce above
end

check_leaks()

facts("test mapreducdim, reducedim on DArrays") do
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
        @fact mapreducedim(t -> t*t, +, DA, dms) --> roughly(mapreducedim(t -> t*t, +, A, dms))
        @fact mapreducedim(t -> t*t, +, DA, dms, 1.0) --> roughly(mapreducedim(t -> t*t, +, A, dms, 1.0))
        @fact reducedim(*, DA, dms) --> roughly(reducedim(*, A, dms))
        @fact reducedim(*, DA, dms, 2.0) --> roughly(reducedim(*, A, dms, 2.0))
    end
    close(DA)
    darray_closeall()   # temp created by the mapreduce above
end

check_leaks()

facts("test statistical functions on DArrays") do
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    context("test mean") do
        for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
            @fact mean(DA,dms) --> roughly(mean(A,dms), atol=1e-12)
        end
    end

    context("test std") do
        for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
            @pending std(DA,dms) --> roughly(std(A,dms), atol=1e-12)
        end
    end
    close(DA)
    darray_closeall()   # temporaries created above
end

check_leaks()

facts("test sum on DArrays") do
    A = randn(100,100)
    DA = distribute(A)

    @fact_throws ArgumentError sum(DA,-1)
    @fact_throws ArgumentError sum(DA, 0)

    @fact sum(DA) --> roughly(sum(A), atol=1e-12)
    @fact sum(DA,1) --> roughly(sum(A,1), atol=1e-12)
    @fact sum(DA,2) --> roughly(sum(A,2), atol=1e-12)
    @fact sum(DA,3) --> roughly(sum(A,3), atol=1e-12)
    close(DA)
    darray_closeall()   # temporaries created above
end

check_leaks()

facts("test size on DArrays") do
    A = randn(100,100)
    DA = distribute(A)

    @fact_throws size(DA, 0) # BoundsError
    @fact size(DA,1) --> size(A,1)
    @fact size(DA,2) --> size(A,2)
    @fact size(DA,3) --> size(A,3)
    close(DA)
end

check_leaks()

# test length / endof
facts("test collections API") do
    A = randn(23,23)
    DA = distribute(A)

    context("test length") do
        @fact length(DA) --> length(A)
    end

    context("test endof") do
        @fact endof(DA) --> endof(A)
    end
    close(DA)
end

check_leaks()

facts("test max / min / sum") do
    a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
    d = distribute(a)

    @fact sum(d) --> sum(a)
    @fact maximum(d) --> maximum(a)
    @fact minimum(d) --> minimum(a)
    @fact maxabs(d) --> maxabs(a)
    @fact minabs(d) --> minabs(a)
    @fact sumabs(d) --> sumabs(a)
    @fact sumabs2(d) --> sumabs2(a)
    close(d)
end

check_leaks()

facts("test all / any") do
    a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
    a = [true for i in 1:100]
    d = distribute(a)

    @fact all(d) --> true
    @fact any(d) --> true

    close(d)

    a[50] = false
    d = distribute(a)
    @fact all(d) --> false
    @fact any(d) --> true

    close(d)

    a = [false for i in 1:100]
    d = distribute(a)
    @fact all(d) --> false
    @fact any(d) --> false

    close(d)

    d = dones(10,10)
    @fact all(x-> x>1.0, d) --> false
    @fact all(x-> x>0.0, d) --> true

    close(d)

    a = ones(10,10)
    a[10] = 2.0
    d = distribute(a)
    @fact any(x-> x == 1.0, d) --> true
    @fact any(x-> x == 2.0, d) --> true
    @fact any(x-> x == 3.0, d) --> false

    close(d)
end

check_leaks()

facts("test count" ) do
    a = ones(10,10)
    a[10] = 2.0
    d = distribute(a)

    @fact count(x-> x == 2.0, d) --> 1
    @fact count(x-> x == 1.0, d) --> 99
    @fact count(x-> x == 0.0, d) --> 0

    close(d)
end

check_leaks()

facts("test prod") do
    a = fill(2, 10);
    d = distribute(a);
    @fact prod(d) --> 2^10

    close(d)
end

check_leaks()

facts("test zeros") do
    context("1D dzeros default element type") do
        A = dzeros(10)
        @fact A --> zeros(10)
        @fact eltype(A) --> Float64
        @fact size(A) --> (10,)
        close(A)
    end

    context("1D dzeros with specified element type") do
        A = dzeros(Int, 10)
        @fact A --> zeros(10)
        @fact eltype(A) --> Int
        @fact size(A) --> (10,)
        close(A)
    end

    context("2D dzeros default element type, Dims constuctor") do
        A = dzeros((10,10))
        @fact A --> zeros((10,10))
        @fact eltype(A) --> Float64
        @fact size(A) --> (10,10)
        close(A)
    end

    context("2D dzeros specified element type, Dims constructor") do
        A = dzeros(Int, (10,10))
        @fact A --> zeros(Int, (10,10))
        @fact eltype(A) --> Int
        @fact size(A) --> (10,10)
        close(A)
    end

    context("2D dzeros, default element type") do
        A = dzeros(10,10)
        @fact A --> zeros(10,10)
        @fact eltype(A) --> Float64
        @fact size(A) --> (10,10)
        close(A)
    end

    context("2D dzeros, specified element type") do
        A = dzeros(Int, 10, 10)
        @fact A --> zeros(Int, 10, 10)
        @fact eltype(A) --> Int
        @fact size(A) --> (10,10)
        close(A)
    end
end

check_leaks()

facts("test dones") do
    context("1D dones default element type") do
        A = dones(10)
        @fact A --> ones(10)
        @fact eltype(A) --> Float64
        @fact size(A) --> (10,)
        close(A)
    end

    context("1D dones with specified element type") do
        A = dones(Int, 10)
        @fact eltype(A) --> Int
        @fact size(A) --> (10,)
        close(A)
    end

    context("2D dones default element type, Dims constuctor") do
        A = dones((10,10))
        @fact A --> ones((10,10))
        @fact eltype(A) --> Float64
        @fact size(A) --> (10,10)
        close(A)
    end

    context("2D dones specified element type, Dims constructor") do
        A = dones(Int, (10,10))
        @fact A --> ones(Int, (10,10))
        @fact eltype(A) --> Int
        @fact size(A) --> (10,10)
        close(A)
    end

    context("2D dones, default element type") do
        A = dones(10,10)
        @fact A --> ones(10,10)
        @fact eltype(A) --> Float64
        @fact size(A) --> (10,10)
        close(A)
    end

    context("2D dones, specified element type") do
        A = dones(Int, 10, 10)
        @fact A --> ones(Int, 10, 10)
        @fact eltype(A) --> Int
        @fact size(A) --> (10,10)
        close(A)
    end
end

check_leaks()

facts("test drand") do
    context("1D drand") do
        A = drand(100)
        @fact eltype(A) --> Float64
        @fact size(A) --> (100,)
        @fact all(x-> x >= 0.0 && x <= 1.0, A) --> true
        close(A)
    end

    context("1D drand, specified element type") do
        A = drand(Int, 100)
        @fact eltype(A) --> Int
        @fact size(A) --> (100,)
        close(A)
    end

    context("2D drand, Dims constructor") do
        A = drand((50,50))
        @fact eltype(A) --> Float64
        @fact size(A) --> (50,50)
        @fact all(x-> x >= 0.0 && x <= 1.0, A) --> true
        close(A)
    end

    context("2D drand") do
        A = drand(100,100)
        @fact eltype(A) --> Float64
        @fact size(A) --> (100,100)
        @fact all(x-> x >= 0.0 && x <= 1.0, A) --> true
        close(A)
    end

    context("2D drand, Dims constructor, specified element type") do
        A = drand(Int, (100,100))
        @fact eltype(A) --> Int
        @fact size(A) --> (100,100)
        close(A)
    end

    context("2D drand, specified element type") do
        A = drand(Int, 100, 100)
        @fact eltype(A) --> Int
        @fact size(A) --> (100,100)
        close(A)
    end
end

check_leaks()

facts("test randn") do
    context("1D drandn") do
        A = drandn(100)
        @fact eltype(A) --> Float64
        @fact size(A) --> (100,)
        close(A)
    end

    context("2D drandn, Dims constructor") do
        A = drandn((50,50))
        @fact eltype(A) --> Float64
        @fact size(A) --> (50,50)
        close(A)
    end

    context("2D drandn") do
        A = drandn(100,100)
        @fact eltype(A) --> Float64
        @fact size(A) --> (100,100)
        close(A)
    end
end

check_leaks()

facts("test c/transpose") do
    context("test ctranspose real") do
        A = drand(Float64, 100, 200)
        @fact A' --> Array(A)'
        close(A)
    end
    context("test ctranspose complex") do
        A = drand(Complex128, 200, 100)
        @fact A' --> Array(A)'
        close(A)
    end
    context("test transpose real") do
        A = drand(Float64, 200, 100)
        @fact A.' --> Array(A).'
        close(A)
    end
    context("test ctranspose complex") do
        A = drand(Complex128, 100, 200)
        @fact A.' --> Array(A).'
        close(A)
    end

    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test convert from subdarray") do
    a = drand(20, 20);

    s = view(a, 1:5, 5:8)
    @fact isa(s, SubDArray) --> true
    @fact s --> convert(DArray, s)

    s = view(a, 6:5, 5:8)
    @fact isa(s, SubDArray) --> true
    @fact s --> convert(DArray, s)
    close(a)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test scalar math") do
    a = drand(20, 20);
    b = convert(Array, a)
    for f in (:abs, :abs2, :acos, :acosd, :acot,
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
        context("$f") do
            @fact (eval(f))(a) --> (eval(f))(b)
        end
    end
    a = a + 1
    b = b + 1
    for f in (:asec, :asecd, :acosh, :acsc, :acscd, :acoth)
        context("$f") do
            @fact (eval(f))(a) --> (eval(f))(b)
        end
    end
    close(a)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

# The mapslices tests have been taken from Base.
# Commented out tests that need to be enabled in due course when DArray support is more complete
facts("test mapslices") do
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
            @fact h[:,i] --> hist(a[:,i],0:0.1:1)[2]
        else
            @fact h[:,i] --> fit(Histogram, a[:,i],0:0.1:1).weights
        end
#        @fact vec(H[i,:]) => hist(vec(a[i,:]),0:0.1:1)[2]
#        @fact s[:,i] => sort(a[:,i])
#        @fact vec(S[i,:]) => sort(vec(a[i,:]))
    end

    # issue #3613
    b = mapslices(sum, dones(Float64, (2,3,4), workers(), [1,1,min(nworkers(),4)]), [1,2])
    @fact size(b) --> exactly((1,1,4))
    @fact all(b.==6) --> true

    # issue #5141
    ## Update Removed the version that removes the dimensions when dims==1:ndims(A)
    c1 = mapslices(x-> maximum(-x), a, [])
#    @fact c1 => -a

    # issue #5177
    c = dones(Float64, (2,3,4,5), workers(), [1,1,1,min(nworkers(),5)])
    m1 = mapslices(x-> ones(2,3), c, [1,2])
    m2 = mapslices(x-> ones(2,4), c, [1,3])
    m3 = mapslices(x-> ones(3,4), c, [2,3])
    @fact size(m1) == size(m2) == size(m3) == size(c) --> true

    n1 = mapslices(x-> ones(6), c, [1,2])
    n2 = mapslices(x-> ones(6), c, [1,3])
    n3 = mapslices(x-> ones(6), c, [2,3])
    n1a = mapslices(x-> ones(1,6), c, [1,2])
    n2a = mapslices(x-> ones(1,6), c, [1,3])
    n3a = mapslices(x-> ones(1,6), c, [2,3])
    @fact (size(n1a) == (1,6,4,5) && size(n2a) == (1,3,6,5) && size(n3a) == (2,1,6,5)) --> true
    @fact (size(n1) == (6,1,4,5) && size(n2) == (6,3,1,5) && size(n3) == (2,6,1,5)) --> true
    close(a)
    close(c)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test scalar ops") do
    a = drand(20,20)
    b = convert(Array, a)
    c = drand(20,20)
    d = convert(Array, c)

    for f in (:+, :-, :.+, :.-, :.*, :./, :.%, :div, :mod)
        context("$f") do
            x = rand()
            @fact (eval(f))(a, x) --> (eval(f))(b, x)
            @fact (eval(f))(x, a) --> (eval(f))(x, b)
            @fact (eval(f))(a, c) --> (eval(f))(b, d)
        end
    end

    close(a)
    close(c)

    a = dones(Int, 20, 20)
    b = convert(Array, a)
    for f in (:.<<, :.>>)
        context("$f") do
            @fact (eval(f))(a, 2) --> (eval(f))(b, 2)
            @fact (eval(f))(2, a) --> (eval(f))(2, b)
            @fact (eval(f))(a, a) --> (eval(f))(b, b)
        end
    end

    for f in (:rem,)
        context("$f") do
            x = rand()
            @fact (eval(f))(a, x) --> (eval(f))(b, x)
        end
    end
    close(a)
    close(c)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test broadcast ops") do
    wrkrs = workers()
    nwrkrs = length(wrkrs)
    nrows = 20 * nwrkrs
    ncols = 10 * nwrkrs
    a = drand((nrows,ncols), wrkrs, (1, nwrkrs))
    m = mean(a, 1)
    c = a .- m
    d = convert(Array, a) .- convert(Array, m)
    @fact c --> d
    darray_closeall()
end

check_leaks()

facts("test matrix multiplication") do
    A = drandn(20,20)
    b = drandn(20)
    B = drandn(20,20)

    @fact norm(convert(Array, A*b) - convert(Array, A)*convert(Array, b), Inf) --> less_than(sqrt(eps()))
    @fact norm(convert(Array, A*B) - convert(Array, A)*convert(Array, B), Inf) -->  less_than(sqrt(eps()))
    @fact norm(convert(Array, A'*b) - convert(Array, A)'*convert(Array, b), Inf) --> less_than(sqrt(eps()))
    @fact norm(convert(Array, A'*B) - convert(Array, A)'*convert(Array, B), Inf) --> less_than(sqrt(eps()))
    close(A)
    close(b)
    close(B)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test norm") do
    x = drandn(20)

    @fact abs(norm(x) - norm(convert(Array, x))) --> less_than(sqrt(eps()))
    @fact abs(norm(x, 1) - norm(convert(Array, x), 1)) --> less_than(sqrt(eps()))
    @fact abs(norm(x, 2) - norm(convert(Array, x), 2)) --> less_than(sqrt(eps()))
    @fact abs(norm(x, Inf) - norm(convert(Array, x), Inf)) --> less_than(sqrt(eps()))
    close(x)
end

check_leaks()

facts("test axpy!") do
    x = drandn(20)
    y = drandn(20)

    @fact norm(convert(Array, LinAlg.axpy!(2.0, x, copy(y))) - LinAlg.axpy!(2.0, convert(Array, x), convert(Array, y))) --> less_than(sqrt(eps()))
    @fact_throws LinAlg.axpy!(2.0, x, zeros(length(x) + 1)) DimensionMismatch
    close(x)
    close(y)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test ppeval") do
    A = drandn((10, 10, nworkers()), workers(), [1, 1, nworkers()])
    B = drandn((10, nworkers()), workers(), [1, nworkers()])

    R = zeros(10, nworkers())
    for i = 1:nworkers()
        R[:, i] = convert(Array, A)[:, :, i]*convert(Array, B)[:, i]
    end
    @fact convert(Array, ppeval(*, A, B)) --> roughly(R)
    @fact sum(ppeval(eigvals, A)) --> roughly(sum(ppeval(eigvals, A, eye(10, 10))))
    close(A)
    close(B)
    darray_closeall()  # close the temporaries created above
end

check_leaks()

facts("test nnz") do
    A = sprandn(10, 10, 0.5)
    @fact nnz(distribute(A)) --> nnz(A)
end

facts("test matmatmul") do
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

    @fact AB --> roughly(ab)
    @fact AtB --> roughly(atb)
    @fact AcB --> roughly(acb)
    darray_closeall()  # close the temporaries created above
end

facts("sort") do
    for i in 1:6
        for T in [Int, Float64]
            d=DistributedArrays.drand(T, 10^i)
            d2=DistributedArrays.samplesort(d)

            @fact length(d) --> length(d2)
            @fact sort(convert(Array, d)) --> convert(Array, d2)
        end
    end
    darray_closeall()  # close the temporaries created above
end

check_leaks()

darray_closeall()

facts("test for any leaks") do
    sleep(1.0)     # allow time for any cleanup to complete
    allrefszero = Bool[remotecall_fetch(()->length(DistributedArrays.refs) == 0, p) for p in procs()]
    @fact all(allrefszero) --> true

    allregistrieszero = Bool[remotecall_fetch(()->length(DistributedArrays.registry) == 0, p) for p in procs()]
    @fact all(allregistrieszero) --> true
end
