Distributed Arrays
------------------

Large computations are often organized around large arrays of data. In
these cases, a particularly natural way to obtain parallelism is to
distribute arrays among several processes. This combines the memory
resources of multiple machines, allowing use of arrays too large to fit
on one machine. Each process operates on the part of the array it
owns, providing a ready answer to the question of how a program should
be divided among machines.

Julia distributed arrays are implemented by the :class:`DArray` type. A
:class:`DArray` has an element type and dimensions just like an :class:`Array`.
A :class:`DArray` can also use arbitrary array-like types to represent the local
chunks that store actual data. The data in a :class:`DArray` is distributed by
dividing the index space into some number of blocks in each dimension.

Common kinds of arrays can be constructed with functions beginning with
``d``::

    dzeros(100,100,10)
    dones(100,100,10)
    drand(100,100,10)
    drandn(100,100,10)
    dfill(x,100,100,10)

In the last case, each element will be initialized to the specified
value ``x``. These functions automatically pick a distribution for you.
For more control, you can specify which processes to use, and how the
data should be distributed::

    dzeros((100,100), workers()[1:4], [1,4])

The second argument specifies that the array should be created on the first
four workers. When dividing data among a large number of processes,
one often sees diminishing returns in performance. Placing :class:`DArray`\ s
on a subset of processes allows multiple :class:`DArray` computations to
happen at once, with a higher ratio of work to communication on each
process.

The third argument specifies a distribution; the nth element of
this array specifies how many pieces dimension n should be divided into.
In this example the first dimension will not be divided, and the second
dimension will be divided into 4 pieces. Therefore each local chunk will be
of size ``(100,25)``. Note that the product of the distribution array must
equal the number of processes.

:func:`distribute(a::Array) <distribute>` converts a local array to a distributed array.

:func:`localpart(a::DArray) <localpart>` obtains the locally-stored portion
of a :class:`DArray`.

:func:`localindexes(a::DArray) <localindexes>` gives a tuple of the index ranges owned by the
local process.

:func:`convert(Array, a::DArray) <convert>` brings all the data to the local process.

Indexing a :class:`DArray` (square brackets) with ranges of indexes always
creates a :class:`SubArray`, not copying any data.


Constructing Distributed Arrays
-------------------------------

The primitive :func:`DArray <DArray>` constructor has the following somewhat elaborate signature::

    DArray(init, dims[, procs, dist])

``init`` is a function that accepts a tuple of index ranges. This function should
allocate a local chunk of the distributed array and initialize it for the specified
indices. ``dims`` is the overall size of the distributed array.
``procs`` optionally specifies a vector of process IDs to use.
``dist`` is an integer vector specifying how many chunks the
distributed array should be divided into in each dimension.

The last two arguments are optional, and defaults will be used if they
are omitted.

As an example, here is how to turn the local array constructor :func:`fill`
into a distributed array constructor::

    dfill(v, args...) = DArray(I->fill(v, map(length,I)), args...)

In this case the ``init`` function only needs to call :func:`fill` with the
dimensions of the local piece it is creating.

Distributed Array Operations
----------------------------

At this time, distributed arrays do not have much functionality. Their
major utility is allowing communication to be done via array indexing, which
is convenient for many problems. As an example, consider implementing the
"life" cellular automaton, where each cell in a grid is updated according
to its neighboring cells. To compute a chunk of the result of one iteration,
each process needs the immediate neighbor cells of its local chunk. The
following code accomplishes this::

    function life_step(d::DArray)
        DArray(size(d),procs(d)) do I
            top   = mod(first(I[1])-2,size(d,1))+1
            bot   = mod( last(I[1])  ,size(d,1))+1
            left  = mod(first(I[2])-2,size(d,2))+1
            right = mod( last(I[2])  ,size(d,2))+1

            old = Array(Bool, length(I[1])+2, length(I[2])+2)
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

As you can see, we use a series of indexing expressions to fetch
data into a local array ``old``. Note that the ``do`` block syntax is
convenient for passing ``init`` functions to the :class:`DArray` constructor.
Next, the serial function ``life_rule`` is called to apply the update rules
to the data, yielding the needed :class:`DArray` chunk. Nothing about ``life_rule``
is :class:`DArray`\ -specific, but we list it here for completeness::

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
