using BinaryCommitteeMachineRSGD
using Base.Test

srand(1)
ok, epochs, minerr = replicatedSGD(Patterns(321, 401), K=5, y=7, batch=80, λ=0.75, γ=0.05, γstep=0.001, formula=:simple, max_epochs=10_000)
@test ok

srand(1)
ok, epochs, minerr = replicatedSGD(Patterns(321, 401), K=5, y=7, batch=80, λ=0.75, γ=0.05, γstep=0.001, formula=:hard, max_epochs=10_000)
@test ok

srand(1)
ok, epochs, minerr = replicatedSGD(Patterns(321, 401), K=5, y=7, batch=80, λ=0.75, γ=0.05, γstep=0.001, formula=:corrected, max_epochs=10_000)
@test ok

srand(1)
ok, epochs, minerr = replicatedSGD(Patterns(321, 401), K=5, y=7, batch=80, λ=0.75, γ=0.05, γstep=0.001, formula=:continuous, max_epochs=10_000)
@test ok

outfile = tempname()
try
    srand(1)
    ok, epochs, minerr = replicatedSGD(Patterns(321, 401), K=5, y=7, batch=80, λ=0.75, γ=0.05, γstep=0.001, formula=:hard, max_epochs=10_000,
                                       quiet=true, outfile=outfile)
    @test ok
finally
    isfile(outfile) && rm(outfile)
end

