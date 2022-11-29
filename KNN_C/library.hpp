/**
* This file is part of Vegetto.

* Vegetto is free software: you can redistribute it and/or modify it under the
* terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.

* Vegetto is distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
* A PARTICULAR PURPOSE. See the GNU General Public License for more details.

* You should have received a copy of the GNU General Public License along with
* Vegetto. If not, see <http://www.gnu.org/licenses/>.

* This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
* de Ciencia, Innovación y Universidades"), and by the European Regional
* Development Fund (ERDF).
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <sys/time.h>
#include <math.h>

#include "npy.hpp"

using namespace std;
namespace py = pybind11;

typedef struct Sample
{
    int val;     // Group of point
    double *data;     // Co-ordinate of point
    float distance; // Distance from test point
} Sample;

typedef struct Dataset
{
    Sample *samples;
    int number_of_samples;
    int classes;
} Dataset;

class Configuration {
    public:
        int features;
        int sub_populations;
        int evaluation_version;
        int k;
        string filename;
        int classes;
        int decision_features;

        Configuration(int features_, int sub_populations_, int evaluation_version_, int k_, string filename_, int classes_, float decision_features_) :
        features(features_), sub_populations(sub_populations_), evaluation_version(evaluation_version_), k(k_),
        filename(filename_), classes(classes_), decision_features(decision_features_) {};

        void set_features(int value){
            features = value;
        }

        int get_features() {
            return features;
        }

        void set_sub_populations(int value){
            sub_populations = value;
        }

        int get_sub_populations() {
            return sub_populations;
        }

        void set_evaluation_version(int value){
            evaluation_version = value;
        }

        int get_evaluation_version() {
            return evaluation_version;
        }

        void set_k(int value){
            k = value;
        }

        int get_k() {
            return k;
        }

        void set_filename(string value){
            filename = value;
        }

        string get_filename() {
            return filename;
        }

        void set_classes(int value){
            classes = value;
        }

        int get_classes() {
            return classes;
        }

        void set_decision_features(float value){
            decision_features = value;
        }

        float get_decision_features() {
            return decision_features;
        }
};

class Individual {
	public:
	    std::vector<int> chromosome;
	    std::vector<float> fitness;

		Individual(std::vector<int> chromosome_,std::vector<float> fitness_) :
		chromosome(chromosome_), fitness(fitness_) {};

		void set_chromosome(std::vector<int> value) {
            chromosome = value;
	  	}

	  	std::vector<int> get_chromosome() {
            return chromosome;
	  	}

	  	void set_fitness(std::vector<float> value) {
            fitness = value;
	  	}

	  	std::vector<float> get_fitness() {
            return fitness;
	  	}
};


class Population {
    public:
        std::vector<Individual> individuals;

		Population(std::vector<Individual> individuals_) : individuals(individuals_) {};

		std::vector<std::vector<int>> evaluate(Configuration config);

		void set_individuals(std::vector<Individual> value) {
            individuals = value;
	  	}

	  	std::vector<Individual> get_individuals() {
            return individuals;
	  	}
};
