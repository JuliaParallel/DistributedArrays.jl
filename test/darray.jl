if nworkers() < 3
    remotecall_fetch(1, () -> addprocs(3))
end

id_me = myid()
id_other = filter(x -> x != id_me, procs())[rand(1:(nprocs()-1))]

d = drand((200,200), [id_me, id_other])
dc = copy(d)

@test d == dc # Should be identical
@spawnat id_other localpart(dc)[1] = 0
@test fetch(@spawnat id_other localpart(d)[1] != 0) # but not point to the same memory

s = convert(Matrix{Float64}, d[1:150, 1:150])
a = convert(Matrix{Float64}, d)
@test a[1:150,1:150] == s

@test fetch(@spawnat id_me localpart(d)[1,1]) == d[1,1]
@test fetch(@spawnat id_other localpart(d)[1,1]) == d[1,101]

d = DArray(I->fill(myid(), map(length,I)), (10,10), [id_me, id_other])
d2 = map(x->1, d)
@test reduce(+, d2) == 100

@test reduce(+, d) == ((50*id_me) + (50*id_other))
map!(x->1, d)
@test reduce(+, d) == 100

let a = randn(10,10)
    da = distribute(a)
    @test scale!(da, 2) == scale!(a, 2)
end

# Test mapreduce on DArrays
let
    # Test that it is functionally equivalent to the standard method
    for _ = 1:25, f = [x -> 2x, x -> x^2, x -> x^2 + 2x - 1], opt = [+, *]
        n = rand(2:50)
        arr = rand(1:100, n)
        darr = distribute(arr)
        @test mapreduce(f, opt, arr) == mapreduce(f, opt, darr)
    end
end

# Test mapreducedim on DArrays
@test mapreducedim(t -> t*t, +, d2, 1) == mapreducedim(t -> t*t, +, convert(Array, d2), 1)
@test mapreducedim(t -> t*t, +, d2, 2) == mapreducedim(t -> t*t, +, convert(Array, d2), 2)
@test mapreducedim(t -> t*t, +, d2, (1,2)) == mapreducedim(t -> t*t, +, convert(Array, d2), (1,2))
dims = (20,20,20)

d = drandn(dims)
da = convert(Array, d)
for dms in (1, 2, 3, (1,2), (1,3), (2,3), (1,2,3))
    @test_approx_eq mapreducedim(t -> t*t, +, d, dms) mapreducedim(t -> t*t, +, da, dms)
    @test_approx_eq mapreducedim(t -> t*t, +, d, dms, 1.0) mapreducedim(t -> t*t, +, da, dms, 1.0)

    @test_approx_eq reducedim(*, d, dms) reducedim(*, da, dms)
    @test_approx_eq reducedim(*, d, dms, 2.0) reducedim(*, da, dms, 2.0)

    # statistical function (works generically through calls to mapreducedim!, i.e. not implemented specifically for DArrays)
    @test_approx_eq mean(d, dms) mean(da, dms)
    # @test_approx_eq std(d, dms) std(da, dms) Requires centralize_sumabs2! for DArrays
end


a = map(x->Int(round(rand() * 100)) - 50, Array(Int, 100,1000))
d = distribute(a)

@test sum(a) == sum(d);
@test maximum(a) == maximum(d);
@test minimum(a) == minimum(d);
@test maxabs(a) == maxabs(d);
@test minabs(a) == minabs(d);
@test sumabs(a) == sumabs(d);
@test sumabs2(a) == sumabs2(d);

a = [true for i in 1:100];
d = distribute(a);

@test all(d) == true
@test any(d) == true

a[50] = false;
d = distribute(a);
@test all(d) == false
@test any(d) == true

a = [false for i in 1:100];
d = distribute(a);
@test all(d) == false
@test any(d) == false

d = dones(10,10);
@test all(x-> x>1.0, d) == false
@test all(x-> x>0.0, d) == true

a = ones(10,10);
a[10] = 2.0;
d = distribute(a);
@test any(x-> x == 1.0, d) == true
@test any(x-> x == 2.0, d) == true
@test any(x-> x == 3.0, d) == false

@test count(x-> x == 2.0, d) == 1
@test count(x-> x == 1.0, d) == 99
@test count(x-> x == 0.0, d) == 0

a = fill(2, 10);
d = distribute(a);
@test prod(d) == 2^10
