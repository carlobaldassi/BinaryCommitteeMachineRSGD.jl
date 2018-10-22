# BinaryCommitteeMachineRSGD.jl documentation

This package implements the Replicated Stochastic Gradient Descent algorithm for
committee machines with binary weights described in the paper
*Unreasonable Effectiveness of Learning Neural Networks: From Accessible States
and Robust Ensembles to Basic Algorithmic Schemes*
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso,
Carlo Lucibello, Luca Saglietti and Riccardo Zecchina,
Proc. Natl. Acad. Sci. U.S.A. 113: E7655-E7662 (2016), [doi:10.1073/pnas.1608103113](http://dx.doi.org/10.1073/pnas.1608103113).

The package requires Julia `0.7` or later.

### Installation

To install the module, use these commands from within Julia:

```
julia> using Pkg

julia> Pkg.clone("https://github.com/carlobaldassi/BinaryCommitteeMachineRSGD.jl")
```

Dependencies will be installed automatically.

### Usage

The module is loaded as any other Julia module:

```
julia> using BinaryCommitteeMachineRSGD
```

The code basically provides a single function which generates a system of interacting
replicated committee machines and tries to learn some patterns. The function and the
patterns constructor are documented below.

```@docs
Patterns
```

```@docs
replicatedSGD
```

