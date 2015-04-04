const MYID = myid()
const OTHERIDS = filter(id-> id != MYID, procs())[rand(1:(nprocs()-1))]

facts("test distribute") do
    A = randn(100,100)

    context("test default distribute") do
        DA = distribute(A)
        @fact length(DA.pmap) => nworkers()
    end

    context("test distribute with procs arguments") do
        DA = distribute(A, procs=[1,2])
        @fact length(DA.pmap) => 2
    end
end

facts("test DArray equality") do
    D = drand((200,200), [MYID, OTHERIDS])
    DC = copy(D)

    context("test isequal(::Array, ::DArray)") do
        @fact D == DC => true
    end

    context("test copy(::DArray) does a copy of each localpart") do
        @spawnat OTHERIDS localpart(DC)[1] = 0
        @fact fetch(@spawnat OTHERIDS localpart(D)[1] != 0) => true
    end
end

facts("test @DArray comprehension constructor") do

    context("test valid use of @DArray") do
        @fact (@DArray [i+j for i=1:10, j=1:10]) => [i+j for i=1:10, j=1:10]
    end

    context("test invalid use of @DArray") do
        @fact_throws ArgumentError eval(:((@DArray [1,2,3,4])))
    end
end

facts("test DArray / Array conversion") do
    D = drand((200,200), [MYID, OTHERIDS])

    context("test convert(::Array, ::(Sub)DArray)") do
        S = convert(Matrix{Float64}, D[1:150, 1:150])
        A = convert(Matrix{Float64}, D)

        @fact A[1:150,1:150] => S
        @fact fetch(@spawnat MYID localpart(D)[1,1]) => D[1,1]
        @fact fetch(@spawnat OTHERIDS localpart(D)[1,1]) => D[1,101]
    end
end

facts("test DArray reduce") do
    D = DArray(id->fill(myid(), map(length,id)), (10,10), [MYID, OTHERIDS])

    context("test reduce") do
        @fact reduce(+, D) => ((50*MYID) + (50*OTHERIDS))
    end

    context("test map / reduce") do
        D2 = map(x->1, D)
        @fact reduce(+, D2) => 100
    end

    context("test map! / reduce") do
        map!(x->1, D)
        @fact reduce(+, D) => 100
    end
end

facts("test scale") do
    A = randn(100,100)
    DA = distribute(A)
    @fact scale!(DA, 2) => scale!(A, 2)
end

facts("test mapreduce on DArrays") do
    for _ = 1:25, f = [x -> 2x, x -> x^2, x -> x^2 + 2x - 1], opt = [+, *]
        A = rand(1:100, rand(2:50))
        DA = distribute(A)
        @fact abs(mapreduce(f, opt, A) - mapreduce(f, opt, DA)) < 1e-12 => true
    end
end

facts("test mapreducedim on DArrays") do
    D = DArray(I->fill(myid(), map(length,I)), (73,73), [MYID, OTHERIDS])
    D2 = map(x->1, D)
    @fact mapreducedim(t -> t*t, +, D2, 1) => mapreducedim(t -> t*t, +, convert(Array, D2), 1)
    @fact mapreducedim(t -> t*t, +, D2, 2) => mapreducedim(t -> t*t, +, convert(Array, D2), 2)
    @fact mapreducedim(t -> t*t, +, D2, (1,2)) => mapreducedim(t -> t*t, +, convert(Array, D2), (1,2))
end

facts("test mapreducdim, reducedim on DArrays") do
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
        @fact mapreducedim(t -> t*t, +, DA, dms) => roughly(mapreducedim(t -> t*t, +, A, dms))
        @fact mapreducedim(t -> t*t, +, DA, dms, 1.0) => roughly(mapreducedim(t -> t*t, +, A, dms, 1.0))

        @fact reducedim(*, DA, dms) => roughly(reducedim(*, A, dms))
        @fact reducedim(*, DA, dms, 2.0) => roughly(reducedim(*, A, dms, 2.0))
    end
end

facts("test statistical functions on DArrays") do
    dims = (20,20,20)
    DA = drandn(dims)
    A = convert(Array, DA)

    context("test mean") do
        for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
            @fact mean(DA,dms) => roughly(mean(A,dms), atol=1e-12)
        end
    end

    context("test std") do
        for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
            @pending std(DA,dms) => roughly(std(A,dms), atol=1e-12)
        end
    end
end

facts("test sum on DArrays") do
    A = randn(100,100)
    DA = distribute(A)

    @fact_throws ArgumentError sum(DA,-1)
    @fact_throws ArgumentError sum(DA, 0)

    @fact sum(DA) => roughly(sum(A), atol=1e-12)
    @fact sum(DA,1) => roughly(sum(A,1), atol=1e-12)
    @fact sum(DA,2) => roughly(sum(A,2), atol=1e-12)
    @fact sum(DA,3) => roughly(sum(A,3), atol=1e-12)
end

facts("test size on DArrays") do
    A = randn(100,100)
    DA = distribute(A)

    @fact_throws size(DA, 0) # BoundsError
    @fact size(DA,1) => size(A,1)
    @fact size(DA,2) => size(A,2)
    @fact size(DA,3) => size(A,3)
end

# test length / endof
facts("test collections API") do
    A = randn(23,23)
    DA = distribute(A)

    context("test length") do
        @fact length(DA) => length(A)
    end

    context("test endof") do
        @fact endof(DA) => endof(A)
    end
end

facts("test max / min / sum") do
    a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
    d = distribute(a)

    @fact sum(d) => sum(a)
    @fact maximum(d) => maximum(a)
    @fact minimum(d) => minimum(a)
    @fact maxabs(d) => maxabs(a)
    @fact minabs(d) => minabs(a)
    @fact sumabs(d) => sumabs(a)
    @fact sumabs2(d) => sumabs2(a)
end

facts("test all / any") do
    a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
    a = [true for i in 1:100]
    d = distribute(a)

    @fact all(d) => true
    @fact any(d) => true

    a[50] = false
    d = distribute(a)
    @fact all(d) => false
    @fact any(d) => true

    a = [false for i in 1:100]
    d = distribute(a)
    @fact all(d) => false
    @fact any(d) => false

    d = dones(10,10)
    @fact all(x-> x>1.0, d) => false
    @fact all(x-> x>0.0, d) => true

    a = ones(10,10)
    a[10] = 2.0
    d = distribute(a)
    @fact any(x-> x == 1.0, d) => true
    @fact any(x-> x == 2.0, d) => true
    @fact any(x-> x == 3.0, d) => false
end

facts("test count" ) do
    a = ones(10,10)
    a[10] = 2.0
    d = distribute(a)

    @fact count(x-> x == 2.0, d) => 1
    @fact count(x-> x == 1.0, d) => 99
    @fact count(x-> x == 0.0, d) => 0
end

facts("test prod") do
    a = fill(2, 10);
    d = distribute(a);
    @fact prod(d) => 2^10
end

facts("test zeros") do
    context("1D dzeros default element type") do
        A = dzeros(10)
        @fact A => zeros(10)
        @fact eltype(A) => Float64
        @fact size(A) => (10,)
    end

    context("1D dzeros with specified element type") do
        A = dzeros(Int, 10)
        @fact A => zeros(10)
        @fact eltype(A) => Int
        @fact size(A) => (10,)
    end

    context("2D dzeros default element type, Dims constuctor") do
        A = dzeros((10,10))
        @fact A => zeros((10,10))
        @fact eltype(A) => Float64
        @fact size(A) => (10,10)
    end

    context("2D dzeros specified element type, Dims constructor") do
        A = dzeros(Int, (10,10))
        @fact A => zeros(Int, (10,10))
        @fact eltype(A) => Int
        @fact size(A) => (10,10)
    end

    context("2D dzeros, default element type") do
        A = dzeros(10,10)
        @fact A => zeros(10,10)
        @fact eltype(A) => Float64
        @fact size(A) => (10,10)
    end

    context("2D dzeros, specified element type") do
        A = dzeros(Int, 10, 10)
        @fact A => zeros(Int, 10, 10)
        @fact eltype(A) => Int
        @fact size(A) => (10,10)
    end
end


facts("test dones") do
    context("1D dones default element type") do
        A = dones(10)
        @fact A => ones(10)
        @fact eltype(A) => Float64
        @fact size(A) => (10,)
    end

    context("1D dones with specified element type") do
        A = dones(Int, 10)
        @fact eltype(A) => Int
        @fact size(A) => (10,)
    end

    context("2D dones default element type, Dims constuctor") do
        A = dones((10,10))
        @fact A => ones((10,10))
        @fact eltype(A) => Float64
        @fact size(A) => (10,10)
    end

    context("2D dones specified element type, Dims constructor") do
        A = dones(Int, (10,10))
        @fact A => ones(Int, (10,10))
        @fact eltype(A) => Int
        @fact size(A) => (10,10)
    end

    context("2D dones, default element type") do
        A = dones(10,10)
        @fact A => ones(10,10)
        @fact eltype(A) => Float64
        @fact size(A) => (10,10)
    end

    context("2D dones, specified element type") do
        A = dones(Int, 10, 10)
        @fact A => ones(Int, 10, 10)
        @fact eltype(A) => Int
        @fact size(A) => (10,10)
    end
end

facts("test drand") do
    context("1D drand") do
        A = drand(100)
        @fact eltype(A) => Float64
        @fact size(A) => (100,)
        @fact all(x-> x >= 0.0 && x <= 1.0, A) => true
    end

    context("2D drand, Dims constructor") do
        A = drand((50,50))
        @fact eltype(A) => Float64
        @fact size(A) => (50,50)
        @fact all(x-> x >= 0.0 && x <= 1.0, A) => true
    end

    context("2D drand") do
        A = drand(100,100)
        @fact eletype(A) => Float64
        @fact size(A) => (100,)
        @fact all(x-> x >= 0.0 && x <= 1.0, A) => true
    end

end

facts("test randn") do
    context("1D drandn") do
        A = drandn(100)
        @fact eltype(A) => Float64
        @fact size(A) => (100,)
    end

    context("2D drandn, Dims constructor") do
        A = drandn((50,50))
        @fact eltype(A) => Float64
        @fact size(A) => (50,50)
    end

    context("2D drandn") do
        A = drandn(100,100)
        @fact eletype(A) => Float64
        @fact size(A) => (100,)
    end
end
