module StatisticsExt

using DistributedArrays: DArray
using Statistics: Statistics

Statistics._mean(f, A::DArray, region) = sum(f, A, dims = region) ./ prod((size(A, i) for i in region))

end
