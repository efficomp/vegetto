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

#include "library.hpp"

Dataset load_dataset(string filename, int classes){
    vector<unsigned long> shape;
	bool fortran_order;
	vector<double> data;
	npy::LoadArrayFromNumpy(filename, shape, fortran_order, data);

    Dataset dataset;

	dataset.samples = new Sample[shape[0]];

	if (shape[0] > shape[1]) {
        for(int i = 0; i < shape[0]; ++i){
            dataset.samples[i].data = new double[shape[1] - 1];
            for(int j = 0; j < shape[1] - 1; ++j){
                int pos = shape[1]*i + j;
                dataset.samples[i].data[j] = data[pos];
            }
            dataset.samples[i].val = data[shape[1]*i + (shape[1] - 1)];
            dataset.samples[i].distance = 0.0;
        }
	}
    else {
        for(int i = 0; i < shape[0]; ++i){
            dataset.samples[i].data = new double[shape[1] - 1];
            for(int j = 0; j < shape[1] - 1; ++j){
                int pos = shape[0]*j + i;
                dataset.samples[i].data[j] = data[pos];
            }
            dataset.samples[i].val = data[((shape[1] - 1) * shape[0]) + i];
            dataset.samples[i].distance = 0.0;
        }
    }

    dataset.number_of_samples = shape[0];
    dataset.classes = classes;

    return dataset;

}

void split_train_test(Dataset dataset, Dataset &data_training_out, Dataset &data_test_out) {
    bool chose_classes[dataset.classes];
    int label;
    vector<int> n_samples, labels_training, labels_test;
    vector<Sample> training;
    vector<Sample> test;

    struct timeval current_time;
	gettimeofday(&current_time, NULL);
	srand(((double)current_time.tv_sec+(double)(current_time.tv_usec)/1000000)*1000000);

    for(int c = 0; c < dataset.classes; ++c)
        chose_classes[c] = true;

    for(int s = 0; s < dataset.number_of_samples; ++s)
        n_samples.push_back(s);

    random_shuffle(n_samples.begin(),n_samples.end());

    for(vector<int>::iterator it = n_samples.begin(); it != n_samples.end(); ++it){
        label = dataset.samples[*it].val;
        if(chose_classes[label]){
            training.push_back(dataset.samples[*it]);
            labels_training.push_back(label);
            chose_classes[label] = false;
        }
        else{
            test.push_back(dataset.samples[*it]);
            labels_test.push_back(label);
            chose_classes[label] = true;
        }
    }

    data_training_out.samples = new Sample[training.size()];
    data_test_out.samples = new Sample[test.size()];

    for(int s = 0; s < training.size(); ++s) {
        data_training_out.samples[s] = training[s];
        data_training_out.samples[s].val = labels_training[s];
    }

    data_training_out.classes = dataset.classes;
    data_training_out.number_of_samples = training.size();

    for(int s = 0; s < test.size(); ++s) {
        data_test_out.samples[s] = test[s];
        data_test_out.samples[s].val = labels_test[s];
    }

    data_test_out.classes = dataset.classes;
    data_test_out.number_of_samples = test.size();
}

bool comparison(Sample a, Sample b)
{
    return (a.distance < b.distance);
}

double *prepare_vector_to_vectorization(Dataset dataset, Individual ind){
    const int number_of_features = ind.chromosome.size();
    double *v = new double[dataset.number_of_samples * number_of_features];

    for (int s = 0; s < dataset.number_of_samples; ++s){
        for (int f = 0; f < number_of_features; ++f) {
            v[s*number_of_features + f] = dataset.samples[s].data[ind.chromosome[f]];
        }
    }

    return v;
}

int classify_point_from_scratch_knn(Dataset data_training, int k, Sample test_sample, std::vector<int> chromosome, Configuration config)
{
    double start;
    double end;

    for (int s = 0; s < data_training.number_of_samples; ++s)
    {
        double distance_aux = 0.0;
        for(std::vector<int>::iterator it_f = chromosome.begin(); it_f != chromosome.end(); ++it_f)
                distance_aux += pow((data_training.samples[s].data[*it_f] - test_sample.data[*it_f]), 2);
        data_training.samples[s].distance = sqrt(distance_aux);
    }

    sort(data_training.samples, data_training.samples + data_training.number_of_samples, comparison);

    vector<int> frequencies;

    for(int i = 0; i < data_training.classes; ++i){
        frequencies.push_back(0);
    }

    for (int s = 0; s < k; ++s)
        frequencies[data_training.samples[s].val]++;

    vector<int>::iterator it = max_element(frequencies.begin(), frequencies.end());
    vector<int> delete_tie;
    int chosen_class;

    if(count(frequencies.begin(),frequencies.end(), *it) > 1){
        for(int i = 0; i < frequencies.size(); ++i)
            if (*it == frequencies[i])
                delete_tie.push_back(i);
        chosen_class = delete_tie[rand() % delete_tie.size()];
    }
    else {
        chosen_class = distance(frequencies.begin(), it);
    }

    return chosen_class;
}

int classify_point_from_optimized_knn(Dataset data_training, int k, Sample test_sample, std::vector<int> chromosome, Configuration config)
{
    double start;
    double end;
    vector<Sample> k_data;

    for (int s = 0; s < k; ++s)
    {
        double distance_aux = 0.0;
        for(std::vector<int>::iterator it_f = chromosome.begin(); it_f != chromosome.end(); ++it_f)
                distance_aux += pow((data_training.samples[s].data[*it_f] - test_sample.data[*it_f]), 2);
        data_training.samples[s].distance = sqrt(distance_aux);
        k_data.push_back(data_training.samples[s]);
    }

    sort(k_data.begin(), k_data.end(), comparison);

    for (int s = k; s < data_training.number_of_samples; ++s)
    {
        double distance_aux = 0.0;
        for(std::vector<int>::iterator it_f = chromosome.begin(); it_f != chromosome.end(); ++it_f)
                distance_aux += pow((data_training.samples[s].data[*it_f] - test_sample.data[*it_f]), 2);
        data_training.samples[s].distance = sqrt(distance_aux);
        vector<Sample>::iterator it = upper_bound(k_data.begin(), k_data.end(), data_training.samples[s], comparison);
        if(it != k_data.end()){
            k_data.insert(it, data_training.samples[s]);
            k_data.pop_back();
        }

    }
    vector<int> frequencies;

    for(int i = 0; i < data_training.classes; ++i){
        frequencies.push_back(0);
    }

    for(int s = 0; s < k; ++s)
        frequencies[k_data[s].val]++;

    vector<int>::iterator it = max_element(frequencies.begin(), frequencies.end());
    vector<int> delete_tie;
    int chosen_class;

    if(count(frequencies.begin(),frequencies.end(), *it) > 1){
        for(int i = 0; i < frequencies.size(); ++i)
            if (*it == frequencies[i])
                delete_tie.push_back(i);
        chosen_class = delete_tie[rand() % delete_tie.size()];
    }
    else {
        chosen_class = distance(frequencies.begin(), it);
    }

    return chosen_class;
}

int classify_point_from_vectorization_knn(Dataset data_training, Dataset data_test, Individual ind, int k, double *vector_training, double *vector_test, int init_test_vector)
{
    vector<Sample> k_data;
    int const chromosome_size = ind.chromosome.size();
    for (int s = 0; s < k; ++s)
    {
        double distance_aux = 0.0;
        #pragma omp simd reduction(+:distance_aux)
        for (int f = 0; f < chromosome_size; ++f){
            distance_aux += pow((vector_training[s * chromosome_size + f] - vector_test[init_test_vector * chromosome_size + f]), 2);
        }
        data_training.samples[s].distance = sqrt(distance_aux);
        k_data.push_back(data_training.samples[s]);
    }

    sort(k_data.begin(), k_data.end(), comparison);

    for (int s = k; s < data_training.number_of_samples; ++s)
    {
        double distance_aux = 0.0;
        #pragma omp simd reduction(+:distance_aux)
        for (int f = 0; f < chromosome_size; ++f)
            distance_aux += pow((vector_training[s * chromosome_size + f] - vector_test[init_test_vector * chromosome_size + f]), 2);

        data_training.samples[s].distance = sqrt(distance_aux);
        vector<Sample>::iterator it = upper_bound(k_data.begin(), k_data.end(), data_training.samples[s], comparison);
        if(it != k_data.end()){
            k_data.insert(it, data_training.samples[s]);
            k_data.pop_back();
        }
    }

    vector<int> frequencies;

    for(int i = 0; i < data_training.classes; ++i)
        frequencies.push_back(0);

    for(int s = 0; s < k; ++s)
        frequencies[k_data[s].val]++;

    vector<int>::iterator it = max_element(frequencies.begin(), frequencies.end());
    vector<int> delete_tie;
    int chosen_class;

    if(count(frequencies.begin(),frequencies.end(), *it) > 1){
        for(int i = 0; i < frequencies.size(); ++i)
            if (*it == frequencies[i])
                delete_tie.push_back(i);
        chosen_class = delete_tie[rand() % delete_tie.size()];
    }
    else {
        chosen_class = distance(frequencies.begin(), it);
    }

    return chosen_class;
}

std::vector<std::vector<int>> Population::evaluate(Configuration config){
    Dataset data_training, data_test, dataset;
    double *vector_training;
    double *vector_test;
    dataset = load_dataset(config.filename, config.classes);

    std::vector<float> final_fitness;
    std::vector<std::vector<int>> ind_labels;
    std::vector<double> accumulated_time;
    double start, end;

	if (config.evaluation_version == 2){
	    for(int ind = 0; ind < individuals.size(); ++ind){
            split_train_test(dataset, data_training, data_test);
            std::vector<int> original_labels(data_test.number_of_samples);
            std::vector<int> predict_labels(data_test.number_of_samples);

            for (int sam = 0; sam < data_test.number_of_samples; ++sam){
                int val = classify_point_from_scratch_knn(data_training, int(round(sqrt(data_training.number_of_samples))), data_test.samples[sam], individuals[ind].chromosome, config);
                original_labels[sam] = data_test.samples[sam].val;
                predict_labels[sam] = val;
            }

            ind_labels.push_back(predict_labels);
            ind_labels.push_back(original_labels);

            delete[] data_training.samples;
            delete[] data_test.samples;
        }
	}
	else if (config.evaluation_version == 3){
	    for(int ind = 0; ind < individuals.size(); ++ind){
            split_train_test(dataset, data_training, data_test);
            std::vector<int> original_labels(data_test.number_of_samples);
            std::vector<int> predict_labels(data_test.number_of_samples);
            for (int sam = 0; sam < data_test.number_of_samples; ++sam){
                int val = classify_point_from_optimized_knn(data_training, int(round(sqrt(data_training.number_of_samples))), data_test.samples[sam], individuals[ind].chromosome, config);
                original_labels[sam] = data_test.samples[sam].val;
                predict_labels[sam] = val;
            }

            ind_labels.push_back(predict_labels);
            ind_labels.push_back(original_labels);
            delete[] data_training.samples;
            delete[] data_test.samples;
        }
	}
	else if (config.evaluation_version == 4) {
        for(int ind = 0; ind < individuals.size(); ++ind){
            split_train_test(dataset, data_training, data_test);
            std::vector<int> original_labels(data_test.number_of_samples);
            std::vector<int> predict_labels(data_test.number_of_samples);

            vector_training = prepare_vector_to_vectorization(data_training,individuals[ind]);
            vector_test = prepare_vector_to_vectorization(data_test,individuals[ind]);

            for (int sam = 0; sam < data_test.number_of_samples; ++sam){
                int val = classify_point_from_vectorization_knn(data_training, data_test, individuals[ind], int(round(sqrt(data_training.number_of_samples))), vector_training, vector_test, sam);
                original_labels[sam] = data_test.samples[sam].val;
                predict_labels[sam] = val;
            }

            ind_labels.push_back(predict_labels);
            ind_labels.push_back(original_labels);

            delete[] data_training.samples;
            delete[] data_test.samples;
            delete[] vector_training;
            delete[] vector_test;
        }

	}
	else if (config.evaluation_version == 5) {
        for(int ind = 0; ind < individuals.size(); ++ind){
            split_train_test(dataset, data_training, data_test);
            std::vector<int> original_labels(data_test.number_of_samples);
            std::vector<int> predict_labels(data_test.number_of_samples);

            if (individuals[ind].chromosome.size() > config.decision_features){
                vector_training = prepare_vector_to_vectorization(data_training,individuals[ind]);
                vector_test = prepare_vector_to_vectorization(data_test,individuals[ind]);

                for (int sam = 0; sam < data_test.number_of_samples; ++sam){
                    int val = classify_point_from_vectorization_knn(data_training, data_test, individuals[ind], int(round(sqrt(data_training.number_of_samples))), vector_training, vector_test, sam);
                    original_labels[sam] = data_test.samples[sam].val;
                    predict_labels[sam] = val;
                }
                delete[] vector_training;
                delete[] vector_test;
            }
            else {
                for (int sam = 0; sam < data_test.number_of_samples; ++sam){
                    int val = classify_point_from_optimized_knn(data_training, int(round(sqrt(data_training.number_of_samples))), data_test.samples[sam], individuals[ind].chromosome, config);
                    original_labels[sam] = data_test.samples[sam].val;
                    predict_labels[sam] = val;
                }
            }

            delete[] data_training.samples;
            delete[] data_test.samples;
            ind_labels.push_back(predict_labels);
            ind_labels.push_back(original_labels);
        }
	}

	for (int i = 0; i < dataset.number_of_samples; ++i)
	    delete[] dataset.samples[i].data;
    delete[] dataset.samples;

	return ind_labels;
}

PYBIND11_MODULE(knn_library, module_handle) {
  	module_handle.doc() = "KNN module";

  	module_handle.def("classify_point_from_scratch_knn", &classify_point_from_scratch_knn);
  	module_handle.def("classify_point_from_optimized_knn", &classify_point_from_optimized_knn);
  	module_handle.def("classify_point_from_vectorization_knn", &classify_point_from_vectorization_knn);
  	module_handle.def("prepare_vector_to_vectorization", &prepare_vector_to_vectorization);
  	module_handle.def("comparison", &comparison);
  	module_handle.def("split_train_test", &split_train_test);
  	module_handle.def("load_dataset", &load_dataset);

    py::class_<Configuration>(
			module_handle, "Configuration"
			).def(py::init<int, int, int, int, string, int, float>())
		.def_property("features", &Configuration::get_features, &Configuration::set_features)
		.def_property("sub_populations", &Configuration::get_sub_populations, &Configuration::set_sub_populations)
		.def_property("evaluation_version", &Configuration::get_evaluation_version, &Configuration::set_evaluation_version)
		.def_property("k", &Configuration::get_k, &Configuration::set_k)
		.def_property("filename", &Configuration::get_filename, &Configuration::set_filename)
		.def_property("classes", &Configuration::get_classes, &Configuration::set_classes)
		.def_property("decision_features", &Configuration::get_decision_features, &Configuration::set_decision_features)
		;

  	py::class_<Individual>(
			module_handle, "Individual"
			).def(py::init<std::vector<int>, std::vector<float>>())
		.def_property("chromosome", &Individual::get_chromosome, &Individual::set_chromosome)
		.def_property("fitness", &Individual::get_fitness, &Individual::set_fitness)
		;

	py::class_<Population>(
			module_handle, "Population"
			).def(py::init<std::vector<Individual>>())
		.def_property("individuals", &Population::get_individuals, &Population::set_individuals)
		.def("evaluate", &Population::evaluate)
		;
}
