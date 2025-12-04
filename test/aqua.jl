using DistributedArrays, Test
import Aqua

@testset "Aqua" begin
    Aqua.test_all(DistributedArrays)
end
