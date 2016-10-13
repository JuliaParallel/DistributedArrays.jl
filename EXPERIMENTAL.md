Experimental code to figure out what a good API using the underlying infrastructure of darrays to distribute types other than DArrays can be.

- distribute ranges
- distribute variables which are constant over a period of computation
- distribute one item per worker ( can hold the result of a computation which needn't be an array)
- distributed dictionaries

Ignoring API names and performance considerations for now, some sample code

```
julia> addprocs(2)
2-element Array{Int64,1}:
 2
 3

julia> @everywhere using DistributedArrays
WARNING: replacing module DistributedArrays.
WARNING: replacing module DistributedArrays.

julia> # distribute a unit range - this does not create an darray like
       # the current DArray implementation does. Returns a Distributed
       # type
       d=distribute(1:99)
DistributedArrays.Distributed{UnitRange}((1,1),[2,3],Nullable{UnitRange{T<:Real}}(1:99),Nullable{Array{UnitRange,1}}(UnitRange[1:49,50:99]),0,true)

julia> println(d[DPid(2)])  # Fetch the range on worker 2
1:49

julia> println(d[])         # Fetch the local range
1:0

julia> # Create a new Distributed type
       d2=map(x->x*2, d)
DistributedArrays.Distributed{Any}((1,2),[2,3],Nullable{Any}(),Nullable{Array{Any,1}}(),0,true)

julia> println(d2[DPid(2)]) # Fetch the data from worker 2
[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98]

julia> try println(d2[]); catch e; println(e); end    # This is an error since a localpart cannot be found locally
ErrorException("localpart does not exist")

julia> println(collect(d2; element_type=Int)) # collect all parts of d2
[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,180,182,184,186,188,190,192,194,196,198]

julia> # Distribute variable d3 on all workers
       d3=distribute_constant("Hello")
DistributedArrays.Distributed{String}((1,3),[2,3],Nullable{String}("Hello"),Nullable{Array{String,1}}(),1,true)

julia> println(d3[DPid(2)]) # Value of d3 on worker 2
Hello

julia> println(d3[])        # Value of d3 locally
Hello

julia> @everywhere type Foo
         foo
       end

julia> # Distribute an instance of Foo on all workers
       d4=distribute_constant(Foo(1))
DistributedArrays.Distributed{Foo}((1,4),[2,3],Nullable{Foo}(Foo(1)),Nullable{Array{Foo,1}}(),1,true)

julia> println(d4[DPid(2)])    # Value of d4 on worker 2
Foo(1)

julia> println(d4[])           # Value of d4 locally
Foo(1)

julia> # create a regular distributed array
       d_arr = drand(10,10);

julia> # convert to distributed type
       d_converted = convert(Distributed, d_arr)
DistributedArrays.Distributed{Array{Float64,2}}((1,6),[2,3],Nullable{Array{Float64,2}}(),Nullable{Array{Array{Float64,2},1}}(),0,false)

julia> # Compute sum locally and store it distributed - this is stored as scalar values everywhere
       d5 = map(x->sum(x), d_converted)
DistributedArrays.Distributed{Any}((1,7),[2,3],Nullable{Any}(),Nullable{Array{Any,1}}(),0,true)

julia> println(d5[DPid(2)]) # Value of d5 on worker 2
25.230014303222525

julia> try println(d5[]); catch e; println(e); end        # Value of d5 locally
ErrorException("localpart does not exist")

julia> reduce(+, d5)
54.215170122991424

julia> reduce(+, d_converted)
54.215170122991424

julia> # Distribute a few scalar constants, a Distributed type and a DArray
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
foo (generic function with 1 method)

julia> foo()
10500.0
```
