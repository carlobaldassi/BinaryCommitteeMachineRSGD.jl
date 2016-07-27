# BinaryCommitteeMachineRSGD.jl documentation

This package implements the Replicated Stochastig Gradient Descent algorithm for
committee machines with binary weights described in the paper
[Unreasonable Effectiveness of Learning Neural Nets: Accessible States and Robust Ensembles](http://arxiv.org/abs/1605.06444)
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso, Carlo Lucibello, Luca Saglietti and Riccardo Zecchina.

The package is tested against Julia `0.4` and *current* `0.5-dev` on Linux, OS X, and Windows.

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

