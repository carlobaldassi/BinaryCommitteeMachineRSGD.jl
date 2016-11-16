# BinaryCommitteeMachineRSGD.jl documentation

This package implements the Replicated Stochastic Gradient Descent algorithm for
committee machines with binary weights described in the paper
*Unreasonable Effectiveness of Learning Neural Networks: From Accessible States
and Robust Ensembles to Basic Algorithmic Schemes*
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso,
Carlo Lucibello, Luca Saglietti and Riccardo Zecchina,
Proc. Natl. Acad. Sci. U.S.A. (2016), [doi:10.1073/pnas.1608103113](http://dx.doi.org/10.1073/pnas.1608103113).

The package is tested against Julia `0.4`, `0.5` and *current* `0.6-dev` on Linux, OS X, and Windows.

### Installation

To install the module, use this command from within Julia:

```
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

