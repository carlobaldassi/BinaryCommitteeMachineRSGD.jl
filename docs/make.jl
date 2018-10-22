using Documenter, BinaryCommitteeMachineRSGD

makedocs(
    modules  = [BinaryCommitteeMachineRSGD],
    format   = :html,
    sitename = "BinaryCommitteeMachineRSGD.jl",
    pages    = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo   = "github.com/carlobaldassi/BinaryCommitteeMachineRSGD.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
    julia  = "1.0"
)
