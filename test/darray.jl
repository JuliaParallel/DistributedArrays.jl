const MYID = myid()
const OTHERIDS = filter(id-> id != MYID, procs())[rand(1:(nprocs()-1))]

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


#=
# statistical function (works generically through calls to mapreducedim!,
i.e. not implemented specifically for DArrays)
@test_approx_eq mean(d, dms) mean(da, dms)
# @test_approx_eq std(d, dms) std(da, dms) Requires centralize_sumabs2! for DArrays
end
=#

facts("test sum on DArrays") do
    A = randn(100,100)
    DA = distribute(A)

    #@test_throws BoundsError sum(DA,-1)
    #@fact sum(DA,0) => sum(DA,0)
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
