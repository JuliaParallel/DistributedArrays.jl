@everywhere function spmd_test1()
    barrier(;tag=:b1)

    if myid() == 1
        @assert SPMD.recvfrom(2) == "Hello from 2"
        println("SPMD: Passed send/recv")
    elseif myid() == 2
        data = "Hello from 2"
        sendto(1, data)
    end

    stime = rand(1:5)
#    println("Sleeping for $stime seconds")
    sleep(stime)
    barrier(;tag=:b2)

    bcast_val = nothing
    if myid() == 1
        bcast_val = rand(2)
    end

    bcast_val = bcast(bcast_val, 1)

    if myid() == 1
        @assert bcast_val == SPMD.recvfrom(2)
        println("SPMD: Passed broadcast")
    elseif myid() == 2
        sendto(1, bcast_val)
    end

    barrier()

    scatter_data = nothing
    if myid() == 1
        scatter_data = rand(Int8, nprocs())
    end
    lp = scatter(scatter_data, 1, tag=1)

    if myid() == 1
        @assert scatter_data[2:2] == SPMD.recvfrom(2)
        println("SPMD: Passed scatter 1")
    elseif myid() == 2
        sendto(1, lp)
    end

    scatter_data = nothing
    if myid() == 1
        scatter_data = rand(Int8, nprocs()*2)
    end
    lp = scatter(scatter_data, 1, tag=2)

    if myid() == 1
        @assert scatter_data[3:4] == SPMD.recvfrom(2)
        println("SPMD: Passed scatter 2")
    elseif myid() == 2
        sendto(1, lp)
    end

    gathered_data = gather(myid(), 1, tag=3)
    if myid() == 1
        @assert gathered_data == procs()
        println("SPMD: Passed gather 1")
    end

    gathered_data = gather([myid(), myid()], 1, tag=4)
    if myid() == 1
        @assert gathered_data == [[p,p] for p in procs()]
        println("SPMD: Passed gather 2")
    end
end

spmd(spmd_test1)

# Test running only on the workers using the spmd function.

# define the function everywhere
@everywhere function foo_spmd(d_in, d_out, n)
    pids=sort(vec(procs(d_in)))
    pididx = findfirst(pids, myid())
    mylp = localpart(d_in)
    localsum = 0

    # Have each node exchange data with its neighbors
    n_pididx = pididx+1 > length(pids) ? 1 : pididx+1
    p_pididx = pididx-1 < 1 ? length(pids) : pididx-1

#    println(p_pididx, " p", pids[p_pididx], " ", n_pididx, " p", pids[n_pididx])
#    println(mylp)

    for i in 1:n
        sendto(pids[n_pididx], mylp[2])
        sendto(pids[p_pididx], mylp[1])

        mylp[2] = SPMD.recvfrom(pids[p_pididx])
        mylp[1] = SPMD.recvfrom(pids[n_pididx])

#        println(mylp)

        barrier(;pids=pids)
        localsum = localsum + mylp[1] + mylp[2]
    end

    # finally store the sum in d_out
    d_out[:L] = localsum
end

# run foo_spmd on all workers, many of them, all concurrently using implictly different contexts.
in_arrays = map(x->DArray(I->fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1]), 1:8)
out_arrays = map(x->ddata(), 1:8)

@sync for i in 1:8
    @async spmd(foo_spmd, in_arrays[i], out_arrays[i], nworkers(); pids=workers())
end
for i in 1:8
    @test Any[sum(workers())*2 for i in 1:nworkers()] == gather(out_arrays[i])
end

println("SPMD: Passed testing of spmd function run concurrently")

# run concurrently with explictly different contexts

# define the function everywhere
@everywhere function foo_spmd2(d_in, d_out, n)
    pids=sort(vec(procs(d_in)))
    pididx = findfirst(pids, myid())
    mylp = localpart(d_in)

    # see if we have a value in the local store.
    store = context_local_storage()

    localsum = get!(store, :LOCALSUM, 0)

    # Have each node exchange data with its neighbors
    n_pididx = pididx+1 > length(pids) ? 1 : pididx+1
    p_pididx = pididx-1 < 1 ? length(pids) : pididx-1

    for i in 1:n
        sendto(pids[n_pididx], mylp[2])
        sendto(pids[p_pididx], mylp[1])

        mylp[2] = SPMD.recvfrom(pids[p_pididx])
        mylp[1] = SPMD.recvfrom(pids[n_pididx])

        barrier(;pids=pids)
        localsum = localsum + mylp[1] + mylp[2]
    end

    # finally store the sum in d_out
    d_out[:L] = localsum
    store[:LOCALSUM] = localsum
end


in_arrays = map(x->DArray(I->fill(myid(), (map(length,I)...,)), (nworkers(), 2), workers(), [nworkers(),1]), 1:8)
out_arrays = map(x->ddata(), 1:8)
contexts = map(x->context(workers()), 1:8)

@sync for i in 1:8
    @async spmd(foo_spmd2, in_arrays[i], out_arrays[i], nworkers(); pids=workers(), context=contexts[i])
end
# Second run will add the value stored in the previous run.
@sync for i in 1:8
    @async spmd(foo_spmd2, in_arrays[i], out_arrays[i], nworkers(); pids=workers(), context=contexts[i])
end

for i in 1:8
    @test Any[2*sum(workers())*2 for i in 1:nworkers()] == gather(out_arrays[i])
end

# verify localstores with appropriate context store values exist.
@everywhere begin
    if myid() != 1
        n = 0
        for (k,v) in DistributedArrays.SPMD.map_ctxts
            store = v.store
            localsum = store[:LOCALSUM]
            if localsum != 2*sum(workers())*2
                println("localsum ", localsum, " != $(2*sum(workers())*2)")
                error("localsum mismatch")
            end
            n += 1
        end
        @assert n == 8
    end
end

# close the contexts
foreach(x->close(x), contexts)

# verify that the localstores have been deleted.
@everywhere begin
    @assert isempty(DistributedArrays.SPMD.map_ctxts)
end

println("SPMD: Passed spmd function with explicit context run concurrently")

