var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#DistributedArrays.jl-1",
    "page": "Introduction",
    "title": "DistributedArrays.jl",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Distributed-Arrays-1",
    "page": "Introduction",
    "title": "Distributed Arrays",
    "category": "section",
    "text": "Large computations are often organized around large arrays of data. In these cases, a particularly natural way to obtain parallelism is to distribute arrays among several processes. This combines the memory resources of multiple machines, allowing use of arrays too large to fit on one machine. Each process operates on the part of the array it owns, providing a ready answer to the question of how a program should be divided among machines.Julia distributed arrays are implemented by the DArray type. A DArray has an element type and dimensions just like an Array. A DArray can also use arbitrary array-like types to represent the local chunks that store actual data. The data in a DArray is distributed by dividing the index space into some number of blocks in each dimension.Common kinds of arrays can be constructed with functions beginning with d:dzeros(100,100,10)\ndones(100,100,10)\ndrand(100,100,10)\ndrandn(100,100,10)\ndfill(x,100,100,10)In the last case, each element will be initialized to the specified value x. These functions automatically pick a distribution for you. For more control, you can specify which processes to use, and how the data should be distributed:dzeros((100,100), workers()[1:4], [1,4])The second argument specifies that the array should be created on the first four workers. When dividing data among a large number of processes, one often sees diminishing returns in performance. Placing DArray\\ s on a subset of processes allows multiple DArray computations to happen at once, with a higher ratio of work to communication on each process.The third argument specifies a distribution; the nth element of this array specifies how many pieces dimension n should be divided into. In this example the first dimension will not be divided, and the second dimension will be divided into 4 pieces. Therefore each local chunk will be of size (100,25). Note that the product of the distribution array must equal the number of processes.distribute(a::Array) converts a local array to a distributed array.\nlocalpart(d::DArray) obtains the locally-stored portionof a  DArray.Localparts can be retrived and set via the indexing syntax too.Indexing via symbols is used for this, specifically symbols :L,:LP,:l,:lp which are all equivalent. For example, d[:L] returns the localpart of d while d[:L]=v sets v as the localpart of d.localindices(a::DArray) gives a tuple of the index ranges owned by thelocal process.convert(Array, a::DArray) brings all the data to the local process.Indexing a DArray (square brackets) with ranges of indices always creates a SubArray, not copying any data."
},

{
    "location": "index.html#Constructing-Distributed-Arrays-1",
    "page": "Introduction",
    "title": "Constructing Distributed Arrays",
    "category": "section",
    "text": "The primitive DArray constructor has the following somewhat elaborate signature:DArray(init, dims[, procs, dist])init is a function that accepts a tuple of index ranges. This function should allocate a local chunk of the distributed array and initialize it for the specified indices. dims is the overall size of the distributed array. procs optionally specifies a vector of process IDs to use. dist is an integer vector specifying how many chunks the distributed array should be divided into in each dimension.The last two arguments are optional, and defaults will be used if they are omitted.As an example, here is how to turn the local array constructor fill into a distributed array constructor:dfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)In this case the init function only needs to call fill with the dimensions of the local piece it is creating.DArrays can also be constructed from multidimensional Array comprehensions with the @DArray macro syntax.  This syntax is just sugar for the primitive DArray constructor:julia> [i+j for i = 1:5, j = 1:5]\n5x5 Array{Int64,2}:\n 2  3  4  5   6\n 3  4  5  6   7\n 4  5  6  7   8\n 5  6  7  8   9\n 6  7  8  9  10\n\njulia> @DArray [i+j for i = 1:5, j = 1:5]\n5x5 DistributedArrays.DArray{Int64,2,Array{Int64,2}}:\n 2  3  4  5   6\n 3  4  5  6   7\n 4  5  6  7   8\n 5  6  7  8   9\n 6  7  8  9  10"
},

{
    "location": "index.html#Construction-from-arrays-generated-on-separate-processes-1",
    "page": "Introduction",
    "title": "Construction from arrays generated on separate processes",
    "category": "section",
    "text": "DArrays can also be constructed from arrays that have been constructed on separate processes, as demonstrated below:ras = [@spawnat p rand(30,30) for p in workers()[1:4]]\nras = reshape(ras,(2,2))\nD   = DArray(ras)An alternative syntax is:r1 = DistributedArrays.remotecall(() -> rand(10,10), workers()[1]) \nr2 = DistributedArrays.remotecall(() -> rand(10,10), workers()[2]) \nr3 = DistributedArrays.remotecall(() -> rand(10,10), workers()[3]) \nr4 = DistributedArrays.remotecall(() -> rand(10,10), workers()[4]) \nD  = DArray(reshape([r1 r2 r3 r4], (2,2))) The distribution of indices across workers can be checked with[@fetchfrom p localindices(D) for p in workers()]"
},

{
    "location": "index.html#Distributed-Array-Operations-1",
    "page": "Introduction",
    "title": "Distributed Array Operations",
    "category": "section",
    "text": "At this time, distributed arrays do not have much functionality. Their major utility is allowing communication to be done via array indexing, which is convenient for many problems. As an example, consider implementing the \"life\" cellular automaton, where each cell in a grid is updated according to its neighboring cells. To compute a chunk of the result of one iteration, each process needs the immediate neighbor cells of its local chunk. The following code accomplishes this::function life_step(d::DArray)\n    DArray(size(d),procs(d)) do I\n        top   = mod(first(I[1])-2,size(d,1))+1\n        bot   = mod( last(I[1])  ,size(d,1))+1\n        left  = mod(first(I[2])-2,size(d,2))+1\n        right = mod( last(I[2])  ,size(d,2))+1\n\n        old = Array{Bool}(undef, length(I[1])+2, length(I[2])+2)\n        old[1      , 1      ] = d[top , left]   # left side\n        old[2:end-1, 1      ] = d[I[1], left]\n        old[end    , 1      ] = d[bot , left]\n        old[1      , 2:end-1] = d[top , I[2]]\n        old[2:end-1, 2:end-1] = d[I[1], I[2]]   # middle\n        old[end    , 2:end-1] = d[bot , I[2]]\n        old[1      , end    ] = d[top , right]  # right side\n        old[2:end-1, end    ] = d[I[1], right]\n        old[end    , end    ] = d[bot , right]\n\n        life_rule(old)\n    end\nendAs you can see, we use a series of indexing expressions to fetch data into a local array old. Note that the do block syntax is convenient for passing init functions to the DArray constructor. Next, the serial function life_rule is called to apply the update rules to the data, yielding the needed DArray chunk. Nothing about life_rule is DArray\\ -specific, but we list it here for completeness::function life_rule(old)\n    m, n = size(old)\n    new = similar(old, m-2, n-2)\n    for j = 2:n-1\n        for i = 2:m-1\n            nc = +(old[i-1,j-1], old[i-1,j], old[i-1,j+1],\n                   old[i  ,j-1],             old[i  ,j+1],\n                   old[i+1,j-1], old[i+1,j], old[i+1,j+1])\n            new[i-1,j-1] = (nc == 3 || nc == 2 && old[i,j])\n        end\n    end\n    new\nend"
},

{
    "location": "index.html#Numerical-Results-of-Distributed-Computations-1",
    "page": "Introduction",
    "title": "Numerical Results of Distributed Computations",
    "category": "section",
    "text": "Floating point arithmetic is not associative and this comes up when performing distributed computations over DArrays.  All DArray operations are performed over the localpart chunks and then aggregated. The change in ordering of the operations will change the numeric result as seen in this simple example:julia> addprocs(8);\n\njulia> using DistributedArrays\n\njulia> A = fill(1.1, (100,100));\n\njulia> sum(A)\n11000.000000000013\n\njulia> DA = distribute(A);\n\njulia> sum(DA)\n11000.000000000127\n\njulia> sum(A) == sum(DA)\nfalseThe ultimate ordering of operations will be dependent on how the Array is distributed."
},

{
    "location": "index.html#Garbage-Collection-and-DArrays-1",
    "page": "Introduction",
    "title": "Garbage Collection and DArrays",
    "category": "section",
    "text": "When a DArray is constructed (typically on the master process), the returned DArray objects stores information on how the array is distributed, which procesor holds which indices and so on. When the DArray object on the master process is garbage collected, all particpating workers are notified and localparts of the DArray freed on each worker.Since the size of the DArray object itself is small, a problem arises as gc on the master faces no memory pressure to collect the DArray immediately. This results in a delay of the memory being released on the participating workers.Therefore it is highly recommended to explcitly call close(d::DArray) as soon as user code has finished working with the distributed array.It is also important to note that the localparts of the DArray is collected from all particpating workers when the DArray object on the process creating the DArray is collected. It is therefore important to maintain a reference to a DArray object on the creating process for as long as it is being computed upon.d_closeall() is another useful function to manage distributed memory. It releases all darrays created from the calling process, including any temporaries created during computation."
},

{
    "location": "index.html#Working-with-distributed-non-array-data-(requires-Julia-0.6)-1",
    "page": "Introduction",
    "title": "Working with distributed non-array data (requires Julia 0.6)",
    "category": "section",
    "text": "The function ddata(;T::Type=Any, init::Function=I->nothing, pids=workers(), data::Vector=[]) can be used to created a distributed vector whose localparts need not be Arrays.It returns a DArray{T,1,T}, i.e., the element type and localtype of the array are the same.ddata() constructs a distributed vector of length nworkers() where each localpart can hold any value, initially initialized to nothing.Argument data if supplied is distributed over the pids. length(data) must be a multiple of length(pids). If the multiple is 1, returns a DArray{T,1,T} where T is eltype(data). If the multiple is greater than 1, returns a DArray{T,1,Array{T,1}}, i.e., it is equivalent to calling distribute(data).gather{T}(d::DArray{T,1,T}) returns an Array{T,1} consisting of all distributed elements of dGiven a DArray{T,1,T} object d, d[:L] returns the localpart on a worker. d[i] returns the localpart on the ith worker that d is distributed over."
},

{
    "location": "index.html#SPMD-Mode-(An-MPI-Style-SPMD-mode-with-MPI-like-primitives,-requires-Julia-0.6)-1",
    "page": "Introduction",
    "title": "SPMD Mode (An MPI Style SPMD mode with MPI like primitives, requires Julia 0.6)",
    "category": "section",
    "text": "SPMD, i.e., a Single Program Multiple Data mode is implemented by submodule DistributedArrays.SPMD. In this mode the same function is executed in parallel on all participating nodes. This is a typical style of MPI programs where the same program is executed on all processors. A basic subset of MPI-like primitives are currently supported. As a programming model it should be familiar to folks with an MPI background.The same block of code is executed concurrently on all workers using the spmd function.# define foo() on all workers\n@everywhere function foo(arg1, arg2)\n    ....\nend\n\n# call foo() everywhere using the `spmd` function\nd_in=DArray(.....)\nd_out=ddata()\nspmd(foo,d_in,d_out; pids=workers()) # executes on all workersspmd is defined as spmd(f, args...; pids=procs(), context=nothing)args is one or more arguments to be passed to f. pids identifies the workers that f needs to be run on. context identifies a run context, which is explained later.The following primitives can be used in SPMD mode.sendto(pid, data; tag=nothing) - sends data to pid\nrecvfrom(pid; tag=nothing) - receives data from pid\nrecvfrom_any(; tag=nothing) - receives data from any pid\nbarrier(;pids=procs(), tag=nothing) - all tasks wait and then proceeed\nbcast(data, pid; tag=nothing, pids=procs()) - broadcasts the same data over pids from pid\nscatter(x, pid; tag=nothing, pids=procs()) - distributes x over pids from pid\ngather(x, pid; tag=nothing, pids=procs()) - collects data from pids onto worker pidTag tag should be used to differentiate between consecutive calls of the same type, for example, consecutive bcast calls.spmd and spmd related functions are defined in submodule DistributedArrays.SPMD. You will need to import it explcitly, or prefix functions that can can only be used in spmd mode with SPMD., for example, SPMD.sendto."
},

{
    "location": "index.html#Example-1",
    "page": "Introduction",
    "title": "Example",
    "category": "section",
    "text": "This toy example exchanges data with each of its neighbors n times.using Distributed\nusing DistributedArrays\naddprocs(8)\n@everywhere using DistributedArrays\n@everywhere using DistributedArrays.SPMD\n\nd_in=d=DArray(I->fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1])\nd_out=ddata();\n\n# define the function everywhere\n@everywhere function foo_spmd(d_in, d_out, n)\n    pids = sort(vec(procs(d_in)))\n    pididx = findfirst(isequal(myid()), pids)\n    mylp = d_in[:L]\n    localsum = 0\n\n    # Have each worker exchange data with its neighbors\n    n_pididx = pididx+1 > length(pids) ? 1 : pididx+1\n    p_pididx = pididx-1 < 1 ? length(pids) : pididx-1\n\n    for i in 1:n\n        sendto(pids[n_pididx], mylp[2])\n        sendto(pids[p_pididx], mylp[1])\n\n        mylp[2] = recvfrom(pids[p_pididx])\n        mylp[1] = recvfrom(pids[n_pididx])\n\n        barrier(;pids=pids)\n        localsum = localsum + mylp[1] + mylp[2]\n    end\n\n    # finally store the sum in d_out\n    d_out[:L] = localsum\nend\n\n# run foo_spmd on all workers\nspmd(foo_spmd, d_in, d_out, 10, pids=workers())\n\n# print values of d_in and d_out after the run\nprintln(d_in)\nprintln(d_out)"
},

{
    "location": "index.html#SPMD-Context-1",
    "page": "Introduction",
    "title": "SPMD Context",
    "category": "section",
    "text": "Each SPMD run is implictly executed in a different context. This allows for multiple spmd calls to be active at the same time. A SPMD context can be explicitly specified via keyword arg context to spmd.context(pids=procs()) returns a new SPMD context.A SPMD context also provides a context local storage, a dict, which can be used to store key-value pairs between spmd runs under the same context.context_local_storage() returns the dictionary associated with the context.NOTE: Implicitly defined contexts, i.e., spmd calls without specifying a context create a context which live only for the duration of the call. Explictly created context objects can be released early by calling close(ctxt::SPMDContext). This will release the local storage dictionaries on all participating pids. Else they will be released when the context object is gc\'ed on the node that created it."
},

{
    "location": "index.html#Nested-spmd-calls-1",
    "page": "Introduction",
    "title": "Nested spmd calls",
    "category": "section",
    "text": "As spmd executes the the specified function on all participating nodes, we need to be careful with nesting spmd calls.An example of an unsafe(wrong) way:function foo(.....)\n    ......\n    spmd(bar, ......)\n    ......\nend\n\nfunction bar(....)\n    ......\n    spmd(baz, ......)\n    ......\nend\n\nspmd(foo,....)In the above example, foo, bar and baz are all functions wishing to leverage distributed computation. However, they themselves may be currenty part of a spmd call. A safe way to handle such a scenario is to only drive parallel computation from the master process.The correct way (only have the driver process initiate spmd calls):function foo()\n    ......\n    myid()==1 && spmd(bar, ......)\n    ......\nend\n\nfunction bar()\n    ......\n    myid()==1 && spmd(baz, ......)\n    ......\nend\n\nspmd(foo,....)This is also true of functions which automatically distribute computation on DArrays.function foo(d::DArray)\n    ......\n    myid()==1 && map!(bar, d)\n    ......\nend\nspmd(foo,....)Without the myid() check, the spmd call to foo would execute map! from all nodes, which is not what we probably want.Similarly @everywhere from within a SPMD run should also be driven from the master node only."
},

{
    "location": "api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api.html#DistributedArrays.DArray",
    "page": "API",
    "title": "DistributedArrays.DArray",
    "category": "type",
    "text": "DArray(init, dims, [procs, dist])\n\nConstruct a distributed array.\n\nThe parameter init is a function that accepts a tuple of index ranges. This function should allocate a local chunk of the distributed array and initialize it for the specified indices.\n\ndims is the overall size of the distributed array.\n\nprocs optionally specifies a vector of process IDs to use. If unspecified, the array is distributed over all worker processes only. Typically, when running in distributed mode, i.e., nprocs() > 1, this would mean that no chunk of the distributed array exists on the process hosting the interactive julia prompt.\n\ndist is an integer vector specifying how many chunks the distributed array should be divided into in each dimension.\n\nFor example, the dfill function that creates a distributed array and fills it with a value v is implemented as:\n\nExample\n\ndfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.dfill-Tuple{Any,Tuple{Vararg{Int64,N}} where N,Vararg{Any,N} where N}",
    "page": "API",
    "title": "DistributedArrays.dfill",
    "category": "method",
    "text": " dfill(x, dims, ...)\n\nConstruct a distributed array filled with value x. Trailing arguments are the same as those accepted by DArray.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.distribute-Tuple{AbstractArray,DArray}",
    "page": "API",
    "title": "DistributedArrays.distribute",
    "category": "method",
    "text": "distribute(A, DA)\n\nDistribute a local array A like the distributed array DA.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.distribute-Tuple{AbstractArray}",
    "page": "API",
    "title": "DistributedArrays.distribute",
    "category": "method",
    "text": " distribute(A[; procs, dist])\n\nConvert a local array to distributed.\n\nprocs optionally specifies an array of process IDs to use. (defaults to all workers) dist optionally specifies a vector or tuple of the number of partitions in each dimension\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.dones-Tuple{Tuple{Vararg{Int64,N}} where N,Vararg{Any,N} where N}",
    "page": "API",
    "title": "DistributedArrays.dones",
    "category": "method",
    "text": "dones(dims, ...)\n\nConstruct a distributed array of ones. Trailing arguments are the same as those accepted by DArray.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.drand-Tuple{Any,Tuple{Vararg{Int64,N}} where N,Vararg{Any,N} where N}",
    "page": "API",
    "title": "DistributedArrays.drand",
    "category": "method",
    "text": " drand(dims, ...)\n\nConstruct a distributed uniform random array. Trailing arguments are the same as those accepted by DArray.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.drandn-Tuple{Tuple{Vararg{Int64,N}} where N,Vararg{Any,N} where N}",
    "page": "API",
    "title": "DistributedArrays.drandn",
    "category": "method",
    "text": " drandn(dims, ...)\n\nConstruct a distributed normal random array. Trailing arguments are the same as those accepted by DArray.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.dzeros-Tuple{Tuple{Vararg{Int64,N}} where N,Vararg{Any,N} where N}",
    "page": "API",
    "title": "DistributedArrays.dzeros",
    "category": "method",
    "text": " dzeros(dims, ...)\n\nConstruct a distributed array of zeros. Trailing arguments are the same as those accepted by DArray.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.localindices-Tuple{DArray}",
    "page": "API",
    "title": "DistributedArrays.localindices",
    "category": "method",
    "text": "localindices(d)\n\nA tuple describing the indices owned by the local process. Returns a tuple with empty ranges if no local part exists on the calling process.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.localpart-Tuple{Any}",
    "page": "API",
    "title": "DistributedArrays.localpart",
    "category": "method",
    "text": "localpart(A)\n\nThe identity when input is not distributed\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.localpart-Union{Tuple{DArray{T,N,A}}, Tuple{A}, Tuple{N}, Tuple{T}} where A where N where T",
    "page": "API",
    "title": "DistributedArrays.localpart",
    "category": "method",
    "text": "localpart(d::DArray)\n\nGet the local piece of a distributed array. Returns an empty array if no local part exists on the calling process.\n\nd[:L], d[:l], d[:LP], d[:lp] are an alternative means to get localparts. This syntaxt can also be used for assignment. For example, d[:L]=v will assign v to the localpart of d.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.ppeval-Tuple{Any,Vararg{Any,N} where N}",
    "page": "API",
    "title": "DistributedArrays.ppeval",
    "category": "method",
    "text": " ppeval(f, D...; dim::NTuple)\n\nEvaluates the callable argument f on slices of the elements of the D tuple.\n\nArguments\n\nf can be any callable object that accepts sliced or broadcasted elements of D. The result returned from f must be either an array or a scalar.\n\nD has any number of elements and the elements can have any type. If an element of D is a distributed array along the dimension specified by dim. If an element of D is not distributed, the element is by default broadcasted and applied on all evaluations of f.\n\ndim is a tuple of integers specifying the dimension over which the elements of D is slices. The length of the tuple must therefore be the same as the number of arguments D. By default distributed arrays are slides along the last dimension. If the value is less than or equal to zero the element are broadcasted to all evaluations of f.\n\nResult\n\nppeval returns a distributed array of dimension p+1 where the first p sizes correspond to the sizes of return values of f. The last dimension of the return array from ppeval has the same length as the dimension over which the input arrays are sliced.\n\nExamples\n\naddprocs(Sys.CPU_THREADS)\n\nusing DistributedArrays\n\nA = drandn((10, 10, Sys.CPU_THREADS), workers(), [1, 1, Sys.CPU_THREADS])\n\nppeval(eigvals, A)\n\nppeval(eigvals, A, randn(10,10)) # broadcasting second argument\n\nB = drandn((10, Sys.CPU_THREADS), workers(), [1, Sys.CPU_THREADS])\n\nppeval(*, A, B)\n\n\n\n\n\n"
},

{
    "location": "api.html#Base.sort-Union{Tuple{DArray{T,1,A} where A}, Tuple{T}} where T",
    "page": "API",
    "title": "Base.sort",
    "category": "method",
    "text": "sort(d::DVector; sample=true, kwargs...) -> DVector\n\nSorts and returns a new distributed vector.\n\nThe sorted vector may not have the same distribution as the original.\n\nKeyword argument sample can take values:\n\ntrue: A sample of max size 512 is first taken from all nodes. This is used to balance the distribution of the sorted array on participating workers. Default is true.\nfalse: No sampling is done. Assumes a uniform distribution between min(d) and max(d)\n2-element tuple of the form (min, max): No sampling is done. Assumes a uniform distribution between specified min and max values\nArray{T}: The passed array is assumed to be a sample of the distribution and is used to balance the sorted distribution.\n\nKeyword argument alg takes the same options Base.sort\n\n\n\n\n\n"
},

{
    "location": "api.html#Distributed.procs-Tuple{DArray}",
    "page": "API",
    "title": "Distributed.procs",
    "category": "method",
    "text": "procs(d::DArray)\n\nGet the vector of processes storing pieces of DArray d.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.locate-Tuple{DArray,Vararg{Int64,N} where N}",
    "page": "API",
    "title": "DistributedArrays.locate",
    "category": "method",
    "text": "locate(d::DArray, I::Int...)\n\nDetermine the index of procs(d) that hold element I.\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.makelocal-Union{Tuple{AT}, Tuple{N}, Tuple{DArray{#s44,#s43,AT} where #s43 where #s44,Vararg{Any,N}}} where AT where N",
    "page": "API",
    "title": "DistributedArrays.makelocal",
    "category": "method",
    "text": "makelocal(A::DArray, I...)\n\nEquivalent to Array(view(A, I...)) but optimised for the case that the data is local. Can return a view into localpart(A)\n\n\n\n\n\n"
},

{
    "location": "api.html#DistributedArrays.next_did",
    "page": "API",
    "title": "DistributedArrays.next_did",
    "category": "function",
    "text": "next_did()\n\nProduces an incrementing ID that will be used for DArrays.\n\n\n\n\n\n"
},

{
    "location": "api.html#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "Modules = [DistributedArrays]"
},

]}
