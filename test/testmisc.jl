addprocs(2)

@everywhere using DistributedArrays

# dsitribute a unit range - this does not create an darray like
# the current DArray implementation does. Returns a `Distributed`
# type
d=distribute(1:99)

println(d[DPid(2)])  # Fetch the range on worker 2
println(d[])         # Fetch the local range

# Create a new Distributed type
d2=map(x->x*2, d)

println(d2[DPid(2)]) # Fetch the data from worker 2
try println(d2[]); catch e; println(e); end    # This is an error since a localpart cannot be found locally

println(collect(d2; element_type=Int)) # collect all parts of d2

# Distribute variable d3 on all workers
d3=distribute_constant("Hello")
println(d3[DPid(2)]) # Value of d3 on worker 2
println(d3[])        # Value of d3 locally

@everywhere type Foo
  foo
end

# Distribute an instance of Foo on all workers
d4=distribute_constant(Foo(1))
println(d4[DPid(2)])    # Value of d4 on worker 2
println(d4[])           # Value of d4 locally


# create a regular distributed array
d_arr = drand(10,10);

# convert to distributed type
d_converted = convert(Distributed, d_arr)

# Compute sum locally and store it distributed - this is stored as scalar values everywhere
d5 = map(x->sum(x), d_converted)
println(d5[DPid(2)]) # Value of d5 on worker 2
try println(d5[]); catch e; println(e); end        # Value of d5 locally

reduce(+, d5)

reduce(+, d_converted)

# Distribute a few scalar constants, a Distributed type and a DArray
# Perform distributed computations
# Perform a final reduction
# Wrap everything in a function due to globals not being serialized
function foo()
    i = distribute_constant(42)
    j = distribute_constant(2.5)
    d_arr = dones(10,10)

    # perform a computation, the result is stored on the workers
    # For a darray, [] returns the first element, for a Distributed type
    # it returns the localpart. For a distributed constant it returns the
    # global variable.
    temp_result = distribute_one_per_worker(p->localpart(d_arr) * i[] * j[])

    d_sum = map(sum, temp_result)
    reduce(+, d_sum)
end

foo()

