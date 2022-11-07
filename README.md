# Vegetto

Vegetto is a [DEAP](https://deap.readthedocs.io/en/master/)-based evolutionary procedure
designed to solve multi-objective optimization feature selection problems.

Specifically, a wrapper has been designed where NSGA-II is used as search strategy,
while *k*-NN is used as classification algorithm for the evaluation of potential solutions.

This wrapper is designed with two objectives in mind: to reach solutions as close as
possible to the global optimum and to perform the computation in an efficient way. For
the latter, four efficient versions of *k*-NN have been developed in C++, where the
data conversion between both languages is carried out with the [Pybind11](https://pybind11.readthedocs.io/en/stable/)- library. If maximum
efficiency is desired, the last version of *k*-NN should be chosen, since this is a mechanism
that chooses the most optimal version depending on the number of selected features.

## Documentation

Vegetto is fully documented in its [github-pages](https://efficomp.github.io/vegetto/). You can also generate its docs from the source code. Simply change directory to the `doc` subfolder and type in `make html`, the documentation will be under `build/html`. You will need [Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation.
