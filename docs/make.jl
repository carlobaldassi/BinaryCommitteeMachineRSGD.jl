using Documenter, BinaryCommitteeMachineRSGD

makedocs()

deploydocs(
    deps   = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    repo   = "github.com/carlobaldassi/BinaryCommitteeMachineRSGD.jl.git",
    julia  = "0.5"
)

