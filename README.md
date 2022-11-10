# Vegetto

Vegetto is a [DEAP](https://deap.readthedocs.io/en/master/)-based evolutionary procedure
designed to solve multi-objective optimization feature selection problems.

Specifically, a wrapper has been designed where NSGA-II is used as search strategy,
while *k*-NN is used as classification algorithm for the evaluation of potential solutions.

This wrapper is designed with two objectives in mind: to reach solutions as close as
possible to the global optimum and to perform the computation in an efficient way. For
the latter, four efficient versions of *k*-NN have been developed in C++, where the
data conversion between both languages is carried out with the [Pybind11](https://pybind11.readthedocs.io/en/stable/)- 
library. If maximum efficiency is desired, the last version of *k*-NN should be chosen, since this is a mechanism
that chooses the most optimal version depending on the number of selected features.

## Documentation

Vegetto is fully documented in its [github-pages](https://efficomp.github.io/vegetto/). You can also generate its docs 
from the source code. Simply change directory to the `doc` subfolder and type in `make html`, the documentation will be 
under `build/html`. You will need [Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation.

## Usage

The bash script `main_script.sh` is in charge of launching all wrapper execution. Apart from the installation of the 
libraries found in the `requirements.txt` file, it's necessary to install two additional ones: Pybind11 and MongoDB.

All the hyperparameters configuration can be found in the `config.xml` file. Concretely, these hyperparameters are:

* `FolderDataset` - Folder name where the training and test data are stored. This folder must be located inside `db` and the
dataset has to be split into four different files: training data, training labels, test data, and test labels. These files
have to be in the `.npy` format. In the `db` folder there are some examples of datasets which belongs to the UCI repository.
* `Features` - Number of features of the dataset used to execute the wrapper.
* `Executions` - Number of executions.
* `Individual` - Number of individuals in each subpopulation.
* `GenerationsConvergence` - Interval of generations to be analyzed to check if convergence has been reached.
* `MaximumGenerations` - Maximum number of generations that the wrapper will be executed.
* `SubPopulations` - Number of subpopulations.
* `Migrations` - Number of migrations. If this hyperparameter is set to 0, the number of migrations will depend on the 
number of generations required to reach the convergence.
* `EvaluationVersion` - (1) Sklearn *k*-NN; (2) KNN_O1; (3) KNN_O2; (4) KNN_O3; (5) KNN_O4;
* `FitnessEvolution` - Boolean hyperparameter that allows to store in the database the fitness evolution along generations.
* `PercentageFS` - Percentage of selected features in the individuals of the initial population.
* `AccuracyConvergence` - Threshold to determine the convergence according to the difference between the mean validation 
accuracy of the individuals in the population. If this hyperparameter is set to -1, the number of generations carried out
will be equal to `GenerationsConvergence` * `Migrations + 1`.
* `SDConvergence` - Threshold to determine the convergence according to the standard deviation of the Kappa coefficient of the
individuals in the population.  If this hyperparameter is set to -1, the number of generations carried out
will be equal to `GenerationsConvergence` * `Migrations + 1`
* `K` - Number of neighbors used in the *k*-NN classification. If this hyperparameter is set to -1, K will be equal to 
the square root of the number of training samples.
* `ExperimentName` - Experiment name.
* `CrossoverProbability` - Probability to apply crossover.
* `MutationProbability` - Probability to apply mutation.
* `Grain` - Probability that an individual belonging to the pareto front is selected to become a migrant.
* `Period` - Number of generations elapsed between migration process.
* `DecisionFeatures` - Threshold of number of selected features to choose between KNN_O2 and KNN_O3.
* `ProjectPath` - Project path where vegetto is located.

Finally, to use either *k*-NN versions, it is necessary to compile the C++ code, which is located in the `KNN_C` folder.
The command to be executed is:

`g++ -O2 -Wall -funroll-loops -march=native -fopenmp -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) library.hpp library.cpp -o knn_library$(python3-config --extension-suffix)
`

This command will create a `.so` object that has to be located in the `src` folder.

## Acknowledgments

This work was supported by project *New Computing Paradigms and Heterogeneous Parallel Architectures for High-Performance and Energy Efficiency of Classification and Optimization Tasks on Biomedical Engineering Applications* ([HPEE-COBE](https://atcproyectos.ugr.es/efficomp/research/projects/hpee-cobe/)), with reference PGC2018-098813-B-C31, funded by the Spanish [*Ministerio de Ciencia, Innovación y Universidades*](https://www.ciencia.gob.es/), and by the [European Regional Development Fund (ERDF)](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: right">
  <a href="https://www.ciencia.gob.es/">
    <img src="https://raw.githubusercontent.com/efficomp/culebra/master/doc/source/_static/micinu.png" height="75">
  </a>
  <a href="https://ec.europa.eu/regional_policy/en/funding/erdf/">
    <img src="https://raw.githubusercontent.com/efficomp/culebra/master/doc/source/_static/erdf.png" height="75">
  </a>
</div>


## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.md) © 2020-2021 [EFFICOMP](https://atcproyectos.ugr.es/efficomp/).