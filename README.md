# DistributedArrays

*Distributed arrays for Julia.*

| **Documentation**                                                         | **Build Status**                                              |
|:-------------------------------------------------------------------------:|:-------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] |

## Introduction

`DistributedArrays.jl` uses the stdlib [`Distributed`][distributed-docs] to implement a *Global Array* interface.
A `DArray` is distributed across a set of workers. Each worker can read and write from its local portion of the array and each worker has read-only access to the portions of the array held by other workers.

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add DistributedArrays
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("DistributedArrays")
```

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.**
- [**DEVEL**][docs-dev-url] &mdash; *documentation of the in-development version.*

## Project Status

The package is tested against Julia `0.7`, `1.0` and the nightly builds of the Julia `master` branch on Linux, and macOS.

## Questions and Contributions

Usage questions can be posted on the [Julia Discourse forum][discourse-tag-url] under the `Parallel/Distributed` category, in the #parallel channel of the [Julia Slack](https://julialang.org/community/).

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems. In particular additions to documentation are encouraged!

[contrib-url]: https://juliadocs.github.io/Documenter.jl/latest/man/contributing/
[discourse-tag-url]: https://discourse.julialang.org/c/domain/parallel

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://juliaparallel.github.io/DistributedArrays.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliaparallel.github.io/DistributedArrays.jl/stable

[travis-img]: https://travis-ci.org/JuliaParallel/DistributedArrays.jl.svg?branch=master
[travis-url]: https://travis-ci.org/JuliaParallel/DistributedArrays.jl

[codecov-img]: https://codecov.io/gh/JuliaParallel/DistributedArrays.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaParallel/DistributedArrays.jl

[issues-url]: https://github.com/JuliaParallel/DistributedArrays.jl/issues
[distributed-docs]: https://docs.julialang.org/en/v1/manual/parallel-computing/#Multi-Core-or-Distributed-Processing-1
