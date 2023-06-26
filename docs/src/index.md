# DistributedArrays.jl

```@contents
```

Distributed Arrays
------------------

Large computations are often organized around large arrays of data. In these
cases, a particularly natural way to obtain parallelism is to distribute arrays
among several processes. This combines the memory resources of multiple
machines, allowing use of arrays too large to fit on one machine. Each process
can read and write to the part of the array it owns and has read-only access to
the parts it doesn't own. This provides a ready answer to the question of how a
program should be divided among machines.

Julia distributed arrays are implemented by the `DArray` type. A
`DArray` has an element type and dimensions just like an `Array`.
A `DArray` can also use arbitrary array-like types to represent the local
chunks that store actual data. The data in a `DArray` is distributed by
dividing the index space into some number of blocks in each dimension.

Common kinds of arrays can be constructed with functions beginning with
`d`:

```julia
dzeros(100,100,10)
dones(100,100,10)
drand(100,100,10)
drandn(100,100,10)
dfill(x,100,100,10)
```

In the last case, each element will be initialized to the specified
value `x`. These functions automatically pick a distribution for you.
For more control, you can specify which processes to use, and how the
data should be distributed:

```julia
dzeros((100,100), workers()[1:4], [1,4])
```

The second argument specifies that the array should be created on the first
four workers. When dividing data among a large number of processes,
one often sees diminishing returns in performance. Placing `DArray`s
on a subset of processes allows multiple `DArray` computations to
happen at once, with a higher ratio of work to communication on each
process.

The third argument specifies a distribution; the nth element of
this array specifies how many pieces dimension n should be divided into.
In this example the first dimension will not be divided, and the second
dimension will be divided into 4 pieces. Therefore each local chunk will be
of size `(100,25)`. Note that the product of the distribution array must
equal the number of processes.

* `distribute(a::Array)` converts a local array to a distributed array.

* `localpart(d::DArray)` obtains the locally-stored portion
  of a  `DArray`.

* Localparts can be retrived and set via the indexing syntax too.
  Indexing via symbols is used for this, specifically symbols `:L`,`:LP`,`:l`,`:lp` which
  are all equivalent. For example, `d[:L]` returns the localpart of `d`
  while `d[:L]=v` sets `v` as the localpart of `d`.

* `localindices(a::DArray)` gives a tuple of the index ranges owned by the
  local process.

* `convert(Array, a::DArray)` brings all the data to the local process.

Indexing a `DArray` (square brackets) with ranges of indices always
creates a `SubArray`, not copying any data.


Constructing Distributed Arrays
-------------------------------

The primitive `DArray` constructor has the following somewhat elaborate signature:

```julia
DArray(init, dims[, procs, dist])
```

`init` is a function that accepts a tuple of index ranges. This function should
allocate a local chunk of the distributed array and initialize it for the specified
indices. `dims` is the overall size of the distributed array.
`procs` optionally specifies a vector of process IDs to use.
`dist` is an integer vector specifying how many chunks the
distributed array should be divided into in each dimension.

The last two arguments are optional, and defaults will be used if they
are omitted.

As an example, here is how to turn the local array constructor `fill`
into a distributed array constructor:

```julia
dfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)
```

In this case the `init` function only needs to call `fill` with the
dimensions of the local piece it is creating.

`DArray`s can also be constructed from multidimensional `Array` comprehensions with
the `@DArray` macro syntax.  This syntax is just sugar for the primitive `DArray` constructor:

```julia
julia> [i+j for i = 1:5, j = 1:5]
5x5 Array{Int64,2}:
 2  3  4  5   6
 3  4  5  6   7
 4  5  6  7   8
 5  6  7  8   9
 6  7  8  9  10

julia> @DArray [i+j for i = 1:5, j = 1:5]
5x5 DistributedArrays.DArray{Int64,2,Array{Int64,2}}:
 2  3  4  5   6
 3  4  5  6   7
 4  5  6  7   8
 5  6  7  8   9
 6  7  8  9  10
```

### Construction from arrays generated on separate processes
`DArray`s can also be constructed from arrays that have been constructed on separate processes, as demonstrated below:
```julia
ras = [@spawnat p rand(30,30) for p in workers()[1:4]]
ras = reshape(ras,(2,2))
D   = DArray(ras)
```
An alternative syntax is:
```julia
r1 = DistributedArrays.remotecall(() -> rand(10,10), workers()[1]) 
r2 = DistributedArrays.remotecall(() -> rand(10,10), workers()[2]) 
r3 = DistributedArrays.remotecall(() -> rand(10,10), workers()[3]) 
r4 = DistributedArrays.remotecall(() -> rand(10,10), workers()[4]) 
D  = DArray(reshape([r1 r2 r3 r4], (2,2))) 
```
The distribution of indices across workers can be checked with
```julia
[@fetchfrom p localindices(D) for p in workers()]
```



Distributed Array Operations
----------------------------

At this time, distributed arrays do not have much functionality. Their
major utility is allowing communication to be done via array indexing, which
is convenient for many problems. As an example, consider implementing the
"life" cellular automaton, where each cell in a grid is updated according
to its neighboring cells. To compute a chunk of the result of one iteration,
each process needs the immediate neighbor cells of its local chunk. The
following code accomplishes this:

```julia
function life_step(d::DArray)
    DArray(size(d),procs(d)) do I
        top   = mod(first(I[1])-2,size(d,1))+1
        bot   = mod( last(I[1])  ,size(d,1))+1
        left  = mod(first(I[2])-2,size(d,2))+1
        right = mod( last(I[2])  ,size(d,2))+1

        old = Array{Bool}(undef, length(I[1])+2, length(I[2])+2)
        old[1      , 1      ] = d[top , left]   # left side
        old[2:end-1, 1      ] = d[I[1], left]
        old[end    , 1      ] = d[bot , left]
        old[1      , 2:end-1] = d[top , I[2]]
        old[2:end-1, 2:end-1] = d[I[1], I[2]]   # middle
        old[end    , 2:end-1] = d[bot , I[2]]
        old[1      , end    ] = d[top , right]  # right side
        old[2:end-1, end    ] = d[I[1], right]
        old[end    , end    ] = d[bot , right]

        life_rule(old)
    end
end
```

As you can see, we use a series of indexing expressions to fetch
data into a local array `old`. Note that the `do` block syntax is
convenient for passing `init` functions to the `DArray` constructor.
Next, the serial function `life_rule` is called to apply the update rules
to the data, yielding the needed `DArray` chunk. Nothing about `life_rule`
is `DArray`-specific, but we list it here for completeness:

```julia
function life_rule(old)
    m, n = size(old)
    new = similar(old, m-2, n-2)
    for j = 2:n-1
        for i = 2:m-1
            nc = +(old[i-1,j-1], old[i-1,j], old[i-1,j+1],
                   old[i  ,j-1],             old[i  ,j+1],
                   old[i+1,j-1], old[i+1,j], old[i+1,j+1])
            new[i-1,j-1] = (nc == 3 || nc == 2 && old[i,j])
        end
    end
    new
end
```



Numerical Results of Distributed Computations
---------------------------------------------

Floating point arithmetic is not associative and this comes up
when performing distributed computations over `DArray`s.  All `DArray`
operations are performed over the localparts and then aggregated.
The change in ordering of the operations will change the numeric result as
seen in this simple example:

```julia
julia> addprocs(8);

julia> using DistributedArrays

julia> A = fill(1.1, (100,100));

julia> sum(A)
11000.000000000013

julia> DA = distribute(A);

julia> sum(DA)
11000.000000000127

julia> sum(A) == sum(DA)
false
```

The ultimate ordering of operations will be dependent on how the `Array` is distributed.



Garbage Collection and `DArray`s
------------------------------

When a `DArray` is constructed (typically on the master process), the returned `DArray` objects stores information on how the
array is distributed, which processor holds which indices and so on. When the `DArray` object
on the master process is garbage collected, all participating workers are notified and
localparts of the `DArray` freed on each worker.

Since the size of the `DArray object itself is small, a problem arises as `gc` on the master faces no memory pressure to
collect the `DArray` immediately. This results in a delay of the memory being released on the participating workers.

Therefore it is highly recommended to explicitly call `close(d::DArray)` as soon as user code
has finished working with the distributed array.

It is also important to note that the localparts of the `DArray` is collected from all participating workers
when the `DArray` object on the process creating the `DArray` is collected. It is therefore important to maintain
a reference to a `DArray` object on the creating process for as long as it is being computed upon.

`d_closeall()` is another useful function to manage distributed memory. It releases all `DArrays` created from
the calling process, including any temporaries created during computation.



Working with distributed non-array data (requires Julia 0.6)
------------------------------------------------------------

The function `ddata(;T::Type=Any, init::Function=I->nothing, pids=workers(), data::Vector=[])` can be used
to created a distributed vector whose localparts need not be Arrays.

It returns a `DArray{T,1,T}`, i.e., the element type and localtype of the array are the same.

`ddata()` constructs a distributed vector of length `nworkers()` where each localpart can hold any value,
initially initialized to `nothing`.

Argument `data` if supplied is distributed over the `pids`. `length(data)` must be a multiple of `length(pids)`.
If the multiple is 1, returns a `DArray{T,1,T}` where T is `eltype(data)`. If the multiple is greater than 1,
returns a `DArray{T,1,Array{T,1}}`, i.e., it is equivalent to calling `distribute(data)`.

`gather{T}(d::DArray{T,1,T})` returns an `Array{T,1}` consisting of all distributed elements of `d`.

Given a `DArray{T,1,T}` object `d`, `d[:L]` returns the localpart on a worker. `d[i]` returns the `localpart`
on the ith worker that `d` is distributed over.



SPMD Mode (An MPI Style SPMD mode with MPI like primitives, requires Julia 0.6)
-------------------------------------------------------------------------------
SPMD, i.e., a Single Program Multiple Data mode, is implemented by submodule `DistributedArrays.SPMD`. In this mode the same function is executed in parallel on all participating nodes. This is a typical style of MPI programs where the same program is executed on all processors. A basic subset of MPI-like primitives are currently supported. As a programming model it should be familiar to folks with an MPI background.

The same block of code is executed concurrently on all workers using the `spmd` function.

```julia
# define foo() on all workers
@everywhere function foo(arg1, arg2)
    ....
end

# call foo() everywhere using the `spmd` function
d_in=DArray(.....)
d_out=ddata()
spmd(foo,d_in,d_out; pids=workers()) # executes on all workers
```

`spmd` is defined as `spmd(f, args...; pids=procs(), context=nothing)`

`args` is one or more arguments to be passed to `f`. `pids` identifies the workers
that `f` needs to be run on. `context` identifies a run context, which is explained
later.

The following primitives can be used in SPMD mode.

- `sendto(pid, data; tag=nothing)` - sends `data` to `pid`

- `recvfrom(pid; tag=nothing)` - receives data from `pid`

- `recvfrom_any(; tag=nothing)` - receives data from any `pid`

- `barrier(;pids=procs(), tag=nothing)` - all tasks wait and then proceeed

- `bcast(data, pid; tag=nothing, pids=procs())` - broadcasts the same data over `pids` from `pid`

- `scatter(x, pid; tag=nothing, pids=procs())` - distributes `x` over `pids` from `pid`

- `gather(x, pid; tag=nothing, pids=procs())` - collects data from `pids` onto worker `pid`

Tag `tag` should be used to differentiate between consecutive calls of the same type, for example,
consecutive `bcast` calls.

`spmd` and spmd related functions are defined in submodule `DistributedArrays.SPMD`. You will need to
import it explicitly, or prefix functions that can can only be used in spmd mode with `SPMD.`, for example,
`SPMD.sendto`.



Example
-------

This toy example exchanges data with each of its neighbors `n` times.

```julia
using Distributed
using DistributedArrays
addprocs(8)
@everywhere using DistributedArrays
@everywhere using DistributedArrays.SPMD

d_in=d=DArray(I->fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1])
d_out=ddata();

# define the function everywhere
@everywhere function foo_spmd(d_in, d_out, n)
    pids = sort(vec(procs(d_in)))
    pididx = findfirst(isequal(myid()), pids)
    mylp = d_in[:L]
    localsum = 0

    # Have each worker exchange data with its neighbors
    n_pididx = pididx+1 > length(pids) ? 1 : pididx+1
    p_pididx = pididx-1 < 1 ? length(pids) : pididx-1

    for i in 1:n
        sendto(pids[n_pididx], mylp[2])
        sendto(pids[p_pididx], mylp[1])

        mylp[2] = recvfrom(pids[p_pididx])
        mylp[1] = recvfrom(pids[n_pididx])

        barrier(;pids=pids)
        localsum = localsum + mylp[1] + mylp[2]
    end

    # finally store the sum in d_out
    d_out[:L] = localsum
end

# run foo_spmd on all workers
spmd(foo_spmd, d_in, d_out, 10, pids=workers())

# print values of d_in and d_out after the run
println(d_in)
println(d_out)
```



SPMD Context
------------

Each SPMD run is implicitly executed in a different context. This allows for multiple `spmd` calls to
be active at the same time. A SPMD context can be explicitly specified via keyword arg `context` to `spmd`.

`context(pids=procs())` returns a new SPMD context.

A SPMD context also provides a context local storage, a dict, which can be used to store
key-value pairs between spmd runs under the same context.

`context_local_storage()` returns the dictionary associated with the context.

NOTE: Implicitly defined contexts, i.e., `spmd` calls without specifying a `context` create a context
which live only for the duration of the call. Explicitly created context objects can be released
early by calling `close(ctxt::SPMDContext)`. This will release the local storage dictionaries
on all participating `pids`. Else they will be released when the context object is gc'ed
on the node that created it.



Nested `spmd` calls
-------------------
As `spmd` executes the specified function on all participating nodes, we need to be careful with nesting `spmd` calls.

An example of an unsafe(wrong) way:
```julia
function foo(.....)
    ......
    spmd(bar, ......)
    ......
end

function bar(....)
    ......
    spmd(baz, ......)
    ......
end

spmd(foo,....)
```
In the above example, `foo`, `bar` and `baz` are all functions wishing to leverage distributed computation. However, they themselves may be currently part of a `spmd` call. A safe way to handle such a scenario is to only drive parallel computation from the master process.

The correct way (only have the driver process initiate `spmd` calls):
```julia
function foo()
    ......
    myid()==1 && spmd(bar, ......)
    ......
end

function bar()
    ......
    myid()==1 && spmd(baz, ......)
    ......
end

spmd(foo,....)
```

This is also true of functions which automatically distribute computation on DArrays.
```julia
function foo(d::DArray)
    ......
    myid()==1 && map!(bar, d)
    ......
end
spmd(foo,....)
```
Without the `myid()` check, the `spmd` call to `foo` would execute `map!` from all nodes, which is probably not what we want.

Similarly `@everywhere` from within a SPMD run should also be driven from the master node only.
