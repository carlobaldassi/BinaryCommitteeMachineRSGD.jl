module BCMRSGDTests
using BinaryCommitteeMachineRSGD
using Test
using Random

const N = 321
const M = 201
const seed = 202
const runseed = 101
Random.seed!(seed)
patterns = Patterns(N, M)
const opts = Dict{Symbol,Any}(:K=>5, :y=>7, :batch=>80, :λ=>0.75, :γ=>0.05, :γstep=>0.001, :max_epochs=>10_000, :seed=>runseed)

for formula in [:simple, :hard, :corrected, :continuous]
    ok, epochs, minerr = replicatedSGD(patterns; formula=formula, opts...)
    @test ok
end

opts[:formula] = :hard

ok, epochs, minerr = replicatedSGD(patterns; opts..., init_equal=false)
@test ok

ok, epochs, minerr = replicatedSGD(patterns; opts..., init_equal=false, center=true)
@test ok

outfile = tempname()
isfile(outfile) && rm(outfile)
try
    ok, epochs, minerr = replicatedSGD(patterns; opts..., quiet=true, outfile=outfile)
    @test ok
finally
    isfile(outfile) && rm(outfile)
end

Random.seed!(seed)
patterns = Patterns([randn(N) for μ = 1:M], randn(M))
ok, epochs, minerr = replicatedSGD(patterns; opts...)
@test ok

end # module
