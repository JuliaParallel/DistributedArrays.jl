```
julia> addprocs(4)
4-element Array{Int64,1}:
 2
 3
 4
 5

julia> @everywhere using DistributedArrays
WARNING: replacing module DistributedArrays.
WARNING: replacing module DistributedArrays.
WARNING: replacing module DistributedArrays.
WARNING: replacing module DistributedArrays.

julia> # broadcast data
       @everywhere type Foo
         foo
       end

julia> B_all = bcast(Foo("Hello"))
DistributedArrays.Distributed{Foo,1,Foo,BCast}

julia> # broadcast an array
       B_all_arr = bcast(rand(2))
DistributedArrays.Distributed{Array{Float64,1},1,Array{Float64,1},BCast}

julia> # test whether the data is available on all machines
       for p in workers()
           println("B_all_arr from $p: ", recvfrom(B_all_arr, p))
           println("B_all from $p: ", recvfrom(B_all, p))
       end
B_all_arr from 2: [0.540676,0.877029]
B_all from 2: Foo("Hello")
B_all_arr from 3: [0.540676,0.877029]
B_all from 3: Foo("Hello")
B_all_arr from 4: [0.540676,0.877029]
B_all from 4: Foo("Hello")
B_all_arr from 5: [0.540676,0.877029]
B_all from 5: Foo("Hello")

julia> # NOTE:
       # recvfrom(d, pid) is equivalent to d[DPid(pid)]
       # d[:L] is equivalent to localpart(d)

       # send to a particular process
       S1 = sendto(rand(2), 2)
DistributedArrays.Distributed{Array{Float64,1},1,Array{Float64,1},BCast}

julia> S2 = sendto("World", 3)
DistributedArrays.Distributed{String,1,String,BCast}

julia> println("S1 from 2: ", recvfrom(S1, 2))
S1 from 2: [0.93668,0.850998]

julia> println("S2 from 3: ", recvfrom(S2, 3))
S2 from 3: World

julia> try recvfrom(S1, 3); catch e; println("S1 not found on 3 "); end
S1 not found on 3

julia> try recvfrom(S2, 2); catch e; println("S2 not found on 2 "); end
S2 not found on 2

julia> # scatter
       rv = rand(nworkers())
4-element Array{Float64,1}:
 0.094717
 0.666209
 0.591417
 0.33996

julia> SC1 = scatter(rv)
DistributedArrays.Distributed{Float64,1,Float64,DAny}

julia> for p in workers()
           println("localpart of SC1 from $p: ", SC1[DPid(p)])
       end
localpart of SC1 from 2: 0.0947170069141472
localpart of SC1 from 3: 0.666209107235316
localpart of SC1 from 4: 0.5914166107468701
localpart of SC1 from 5: 0.33996001426693034

julia> G = gather(SC1)
DistributedArrays.Distributed{Array{Float64,1},1,Array{Float64,1},BCast}

julia> println(G[:L])
[0.094717,0.666209,0.591417,0.33996]

julia> @assert G[:L] == rv

julia> AG = all_gather(SC1)
DistributedArrays.Distributed{Array{Float64,1},1,Array{Float64,1},DAny}

julia> for p in workers()
           println("localpart of AG from $p: ", AG[DPid(p)])
       end
localpart of AG from 2: [0.094717,0.666209,0.591417,0.33996]
localpart of AG from 3: [0.094717,0.666209,0.591417,0.33996]
localpart of AG from 4: [0.094717,0.666209,0.591417,0.33996]
localpart of AG from 5: [0.094717,0.666209,0.591417,0.33996]

julia> v = reduce(+, SC1)
1.6923027391632637

julia> println("reduced val : ", v)
reduced val : 1.6923027391632637

julia> RV = all_reduce(+, SC1)
DistributedArrays.Distributed{Float64,1,Float64,BCast}

julia> println(typeof(RV))
DistributedArrays.Distributed{Float64,1,Float64,BCast::DistributedArrays.DISTTYPE = 2}

julia> println(RV[:L])
1.6923027391632637

julia> # MPI all_to_all only on DAny types where the localpart is an array
       d = Distributed(I->[((I*10):(I*10)+7)...]; disttype=DAny)
DistributedArrays.Distributed{Array{Int64,1},1,Array{Int64,1},DAny}

julia> d2 = all_to_all(d)
DistributedArrays.Distributed{Array{Int64,1},1,Array{Int64,1},DAny}

julia> for p in procs(d)
           println(d[DPid(p)]')
       end
[10 11 12 13 14 15 16 17]
[20 21 22 23 24 25 26 27]
[30 31 32 33 34 35 36 37]
[40 41 42 43 44 45 46 47]

julia> for p in procs(d2)
           println(d2[DPid(p)]')
       end
[10 11 20 21 30 31 40 41]
[12 13 22 23 32 33 42 43]
[14 15 24 25 34 35 44 45]
[16 17 26 27 36 37 46 47]

julia> # Differences from MPI
       # - Each of the MPI like calls returns an instance of a Distributed type.
       # - A Distributed type is a handle to the distributed object
       # - There are no global name bindings on any of the workers
       # - The handles need to be passed around
       #   - either captured in closures
       #   - or as explicit arguments to functions

       # Questions:
       # Assigning to a Bcast type will assign everywhere?
       # convert from BCast type to DAny type?
       # scatter of a scalar is a BCast?
       # Or merge BCast and DAny into a DAny type.

```
