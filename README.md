# BinaryCommitteeMachineRSGD.jl

| **Documentation**                       | **Build Status**                                                                                |
|:---------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-latest-img]][docs-latest-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] [![][codecov-img]][codecov-url] |

This package implements the Replicated Stochastic Gradient Descent algorithm for
committee machines with binary weights described in the paper
*Unreasonable Effectiveness of Learning Neural Networks: From Accessible States
and Robust Ensembles to Basic Algorithmic Schemes*
by Carlo Baldassi, Christian Borgs, Jennifer Chayes, Alessandro Ingrosso,
Carlo Lucibello, Luca Saglietti and Riccardo Zecchina,
Proc. Natl. Acad. Sci. U.S.A. (2016), [doi:10.1073/pnas.1608103113](http://dx.doi.org/10.1073/pnas.1608103113).

The code is written in [Julia](http://julialang.org).

The package is tested against Julia `0.4`, `0.5` and *current* `0.6-dev` on Linux, OS X, and Windows.

### Installation

To install the module, use this command from within Julia:

```
julia> Pkg.clone("https://github.com/carlobaldassi/BinaryCommitteeMachineRSGD.jl")
```

Dependencies will be installed automatically.

## Documentation

- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://carlobaldassi.github.io/BinaryCommitteeMachineRSGD.jl/latest

[travis-img]: https://travis-ci.org/carlobaldassi/BinaryCommitteeMachineRSGD.jl.svg?branch=master
[travis-url]: https://travis-ci.org/carlobaldassi/BinaryCommitteeMachineRSGD.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/owdb252j4nob8kya/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/carlobaldassi/binarycommitteemachinersgd-jl/branch/master

[codecov-img]: https://codecov.io/gh/carlobaldassi/BinaryCommitteeMachineRSGD.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/carlobaldassi/BinaryCommitteeMachineRSGD.jl
