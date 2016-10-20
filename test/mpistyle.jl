addprocs(4)

@everywhere using DistributedArrays

# mpi style functions
# bcast, sendto, recvfrom, all_reduce, scatter, gather, all_to_all, all_gather

# NOTE:
# recvfrom(d, pid) is equivalent to d[DPid(pid)]
# d[:L] is equivalent to localpart(d)

# Differences from MPI
# - Each of the MPI like calls returns an instance of a Distributed type.
# - A Distributed type is a handle to the distributed object
# - There are no global name bindings on any of the workers
# - The handles need to be passed around
#   - either captured in closures
#   - or as explicit arguments to functions
#   - only an identity tuple of 2 integers is serialized on the wire.

# Questions:
# Are different BCast and DAny types really required?

# Next steps
# - Remove BCast and DAny, have the older DArray support localparts which are not arrays
# - More mpi style functions - the "v" varieties
# - Translate a typical MPI program into darray with mpi style

# broadcast data
@everywhere type Foo
  foo
end

B_all = bcast(Foo("Hello"))

# broadcast an array
B_all_arr = bcast(rand(2))

# test whether the data is available on all machines
for p in workers()
    println("B_all_arr from $p: ", recvfrom(B_all_arr, p))
    println("B_all from $p: ", recvfrom(B_all, p))
end

# send to a particular process
S1 = sendto(rand(2), 2)
S2 = sendto("World", 3)
println("S1 from 2: ", recvfrom(S1, 2))
println("S2 from 3: ", recvfrom(S2, 3))
try recvfrom(S1, 3); catch e; println("S1 not found on 3 "); end
try recvfrom(S2, 2); catch e; println("S2 not found on 2 "); end

# scatter
rv = rand(nworkers())
SC1 = scatter(rv)
for p in workers()
    println("localpart of SC1 from $p: ", SC1[DPid(p)])
end

G = gather(SC1)
println(G[:L])

@assert G[:L] == rv

AG = all_gather(SC1)
for p in workers()
    println("localpart of AG from $p: ", AG[DPid(p)])
end

v = reduce(+, SC1)
println("reduced val : ", v)
RV = all_reduce(+, SC1)
println(typeof(RV))
println(RV[:L])


# MPI all_to_all only on DAny types where the localpart is an array
d = Distributed(I->[((I*10):(I*10)+7)...]; disttype=DAny)
d2 = all_to_all(d)
for p in procs(d)
    println(d[DPid(p)]')
end
for p in procs(d2)
    println(d2[DPid(p)]')
end


