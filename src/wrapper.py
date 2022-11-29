# This file is part of Vegetto.

# Vegetto is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# Vegetto is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# Vegetto. If not, see <http://www.gnu.org/licenses/>.

# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

import os
import random
import math
import gc
from time import time
import multiprocessing as mp

from deap import base
from deap import creator
from deap import tools
from functools import partial
from sklearn.metrics import cohen_kappa_score

import numpy as np
import statistics as st

from knn import Knn
from config import Config

# from knn_library import *

toolbox = base.Toolbox()


def fill_chromosome(p: float, features: int):
    """
    Fill the chromosome with random features

    :param p: Probability of each feature being selected
    :type float:  :py:class:`float`

    :param features: Number of features of the dataset
    :type int:  :py:class:`int`

    :return Array with selected features
    :rtype: :py:class:`numpy.array`
    """
    aux_features = []

    for i in range(features):
        if random.uniform(0, 1) < p:
            aux_features.append(i)

    return np.array(aux_features)


def random_feature(features: int):
    """
    Generation of an array with a one randome feature

    :param features: Number of features of the dataset
    :type int:  :py:class:`int`

    :return Array with one selected feature
    :rtype: :py:class:`numpy.array`
    """
    return np.array([random.randint(0, features - 1)])


def recombination_and_mutation(pool, config: Config):
    """
    Recombination of the pool's individuals

    :param pool: Pool of individuals

    :param config: Config object where all the hyperparameter values are loaded
    :type Config: :py:mod:`config`

    :return Offspring
    """

    offspring = []
    for parent1, parent2 in zip(pool[::2], pool[1::2]):
        if random.random() < config.pc:
            common = np.intersect1d(parent1, parent2)

            uncommon_parent_1 = np.setdiff1d(parent1, common)
            uncommon_parent_2 = np.setdiff1d(parent2, common)

            random_array_1 = np.random.rand(1, len(uncommon_parent_1))
            random_array_2 = np.random.rand(1, len(uncommon_parent_2))

            uniform_parent_1 = np.concatenate((common, uncommon_parent_1[np.where(random_array_1 < 0.5)[1]],
                                               uncommon_parent_2[np.where(random_array_2 >= 0.5)[1]]))
            uniform_parent_2 = np.concatenate((common, uncommon_parent_1[np.where(random_array_1 >= 0.5)[1]],
                                               uncommon_parent_2[np.where(random_array_2 < 0.5)[1]]))

            pre_child_1 = mutate(uniform_parent_1, config.features, config.pm)
            pre_child_2 = mutate(uniform_parent_2, config.features, config.pm)

            offspring.append(creator.Individual(pre_child_1))
            offspring.append(creator.Individual(pre_child_2))
        else:
            parent_array_1 = np.array(parent1)
            parent_array_2 = np.array(parent2)
            individual_mutated_1 = mutate(parent_array_1, config.features, config.pm)
            individual_mutated_2 = mutate(parent_array_2, config.features, config.pm)

            if not np.array_equal(individual_mutated_1, parent_array_1):
                offspring.append(creator.Individual(individual_mutated_1))
            if not np.array_equal(individual_mutated_2, parent_array_2):
                offspring.append(creator.Individual(individual_mutated_2))

    return offspring


def mutate(individual, features: int, pm: float):
    """
    Mutate to an individual with integer representation

    :param individual: Individual (potential solution)

    :param features: Number of features of the dataset
    :type int:  :py:class:`int`

    :param p: Probability of applying mutation
    :type float:  :py:class:`float`

    :return Individual mutated
    """
    all_features = np.arange(features)
    not_selected = np.setdiff1d(all_features, individual)

    to_be_kept = individual[np.random.random(len(individual)) > pm]

    to_be_added = not_selected[np.random.random(not_selected.size) <= pm]

    return np.sort(np.concatenate((to_be_kept, to_be_added)))


def evaluate_knn_python(individual, knn: Knn):
    """
    Evaluation of an individual by using the Scikit-learn k-NN

    :param individual: Individual (potential solution)

    :param knn: knn object
    :type knn:  :py:class:`Knn`

    :param p: Probability of applying mutation
    :type Knn:  :py:mod:`Knn`

    :return Individual evaluted (fitness assignment)
    """

    knn.calculate_kappa_coefficiente_validation(individual)
    individual.fitness.values = (knn.accuracy_validation, knn.number_of_selected_features)
    return individual


def migration(population, subpops: int, sub_population_size: int):
    """
    Migration between different subpopulations: every subpopulation gives the half of its
    Pareto front to the next subpopulation following a ring topology.

    :param population: Entire population

    :param subpops: Number of subpopulations
    :type int:  :py:class:`int`

    :param sub_population_size: Number of individuals of each subpopulation
    :type int:  :py:class:`int`
    """

    for sub_pop in range(subpops):
        pareto_front = tools.sortNondominated(population[sub_pop], sub_population_size, first_front_only=True)
        if len(pareto_front[0]) > 1:
            dest_sub_pop = 0 if sub_pop == (subpops - 1) else sub_pop + 1
            population[dest_sub_pop] += population[sub_pop][0:math.ceil(len(pareto_front[0]) / 2)]


def evaluation(population, knn: Knn, config: Config):
    """
    Function to choose between the different k-NN implementations

    :param population: Entire population

    :param knn: knn object
    :type knn:  :py:class:`Knn`

    :param config: Config object where all the hyperparameter values are loaded
    :type Config: :py:mod:`config`
    """

    for i in range(len(population)):
        if len(population[i]) == 0:
            population[i] = toolbox.population_for_empty_individual(n=1)[0]
    if config.evaluation_version == 1:
        population = [toolbox.evaluate_knn_python(ind, knn) for ind in population]
    else:
        pop_for_library = Population([Individual(ind, ind.fitness.values) for ind in population])
        config_for_library = Configuration(config.features, config.sub_populations, config.evaluation_version, config.k,
                                           config.project_path + "/db/aux/data_to_c.npy",
                                           len(np.unique(knn.labels_train)), config.decision_features)

        fitness = pop_for_library.evaluate(config_for_library)

        for index in range(len(population)):
            population[index].fitness.values = [cohen_kappa_score(fitness[index * 2], fitness[(index * 2) + 1]),
                                                len(population[index])]


def genetic_algorithm(population, knn: Knn, config: Config, sub_pop: int, q_in, q_out, q, data_queue):
    """
    Implementation of the NSGA-II steps

    :param population: Entire population

    :param knn: knn object
    :type knn:  :py:class:`Knn`

    :param config: Config object where all the hyperparameter values are loaded
    :type Config: :py:mod:`config`

    :param sub_pop: subpopulation index
    :type int:  :py:class:`int`

    :param q_in: queue to put migrants to be send on to next subpopulation
    :type multiprocessing.Queue: :py:mod:`Queue`

    :param q_put: queue to receive migrants
    :type multiprocessing.Queue: :py:mod:`Queue`

    :param q: queue to send the entire population at the end of the wrapper execution
    :type multiprocessing.Queue: :py:mod:`Queue`

    :param data_Qqeue: queue to send relevant information about wrapper execution
    :type multiprocessing.Queue: :py:mod:`Queue`
    """

    tr, te, ts = 0.0, 0.0, 0.0
    m, gen = 0, 0
    finish = False
    tt = time()
    evaluation(population[sub_pop], knn, config)
    te += (time() - tt)

    tt = time()
    population[sub_pop] = toolbox.selectNSGA2(population[sub_pop], len(population[sub_pop]))
    ts += (time() - tt)
    mean_accuracy_convergence, sd_convergence = np.array([]), np.array([])

    accuracy_evolution, features_evolution = np.array([]), np.array([])

    number_of_evaluations = len(population[sub_pop])

    while not finish:
        # -------- Starting the evaluation process --------

        if not q_out.empty():
            pareto_to_pop = q_out.get_nowait()
            population[sub_pop] += pareto_to_pop
            population[sub_pop] = toolbox.selectNSGA2(population[sub_pop], config.individuals)

        # -------- Creation of the pool --------
        tt = time()
        pool = tools.selTournamentDCD(population[sub_pop], config.individuals)
        offspring = recombination_and_mutation(pool, config)
        tr += (time() - tt)

        # ---------------------------------------------- EVALUATION -----------------------------------------------
        tt = time()
        evaluation(offspring, knn, config)
        te += (time() - tt)
        number_of_evaluations += len(offspring)

        # ----------------------------------------------- SELECTION -----------------------------------------------
        tt = time()
        population[sub_pop] = toolbox.selectNSGA2(offspring + population[sub_pop], config.individuals)
        ts += (time() - tt)

        data_ac = [x.fitness.values[0] for x in population[sub_pop]]
        mean_accuracy_convergence = np.append(mean_accuracy_convergence, st.mean(data_ac))
        sd_convergence = np.append(sd_convergence, st.pstdev(data_ac))
        print("Max: ", max(data_ac), "Mean: ", st.mean(data_ac), " SD: ", st.pstdev(data_ac))

        if config.fitness_evolution:
            data_f = [x.fitness.values[1] for x in population[sub_pop]]
            accuracy_evolution = np.append(accuracy_evolution, st.mean(data_ac))
            features_evolution = np.append(features_evolution, st.mean(data_f))
            print("FEATURES Max: ", max(data_f), "Mean: ", st.mean(data_f), " SD: ", st.pstdev(data_f))

        if gen > config.max_generations:
            finish = True
            break
        print(tr, te, ts)
        gc.collect()
        gen += 1

        # ------------------------------------------------- MIGRATION -------------------------------------------------
        if config.sub_populations > 1 and gen % config.period == 0:
            pareto_front = tools.sortNondominated(population[sub_pop], config.individuals, first_front_only=True)[0]
            migration_probability = np.random.rand(1, len(pareto_front))
            individuals_to_migrate = []
            for i in np.where(migration_probability < config.grain)[1]:
                individuals_to_migrate.append(pareto_front[i])
            if len(individuals_to_migrate) == 0:
                individuals_to_migrate.append(pareto_front[random.randint(0, len(pareto_front) - 1)])
            q_in.put_nowait(individuals_to_migrate)
            m += 1
            if m == config.migrations:
                finish = True
        if gen % config.generations_convergence == 0:
            if config.accuracy_convergence == -1 or config.sd_convergence == -1:
                finish = True
            elif (np.max(mean_accuracy_convergence) - np.min(
                    mean_accuracy_convergence)) <= config.accuracy_convergence and len(
                np.argwhere(sd_convergence > config.sd_convergence)) == 0:
                finish = True
            else:
                mean_accuracy_convergence, sd_convergence = np.array([]), np.array([])

    q.put(population[sub_pop])
    if config.fitness_evolution:
        data_queue.put([number_of_evaluations, gen, te, ts, list(accuracy_evolution), list(features_evolution)])
    else:
        data_queue.put([number_of_evaluations, gen, te, ts])


def feature_selection(knn: Knn, config: Config):
    """
    Main function to perform the feature selection

    :param knn: knn object
    :type knn:  :py:class:`Knn`

    :param config: Config object where all the hyperparameter values are loaded
    :type Config: :py:mod:`config`

    :return data_backup : Backup of the wrapper execution data
    :rtype: :py:class:`dict`
    """

    # -----------------
    # Defining genetics operators
    # -----------------

    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    toolbox.register("fill_chromosome", fill_chromosome)

    chromosome_generator = partial(fill_chromosome, config.percentage_fs, config.features)
    empty_chromosome = partial(random_feature, config.features)
    toolbox.register("individual", tools.initIterate, creator.Individual, chromosome_generator)
    toolbox.register("fill_empty_individual", tools.initIterate, creator.Individual, empty_chromosome)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("population_for_empty_individual", tools.initRepeat, list, toolbox.fill_empty_individual)

    toolbox.register("evaluate_knn_python", evaluate_knn_python)

    toolbox.register("crossover", tools.cxUniform, indpb=0.5)

    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    toolbox.register("selectNSGA2", tools.selNSGA2)
    toolbox.register("selectTournament", tools.selTournament, tournsize=2)

    # -----------------
    # Genetic Algorithm
    # -----------------

    random.seed(time())
    np.random.seed(int(time()))

    if config.evaluation_version != 1:
        data_to_c = np.append(knn.data_train, knn.labels_train.reshape(len(knn.labels_train), 1),
                              axis=1)
        np.save(config.project_path + "/db/aux/data_to_c", data_to_c)

    pop = []
    for i in range(config.sub_populations):
        pop.append(toolbox.population(n=config.individuals))

    # ---------------------------------------- QUEUES AND PROCESSES CREATION ------------------------------------------

    processes = []
    final_pop = []
    queues = []
    gen_and_eval = []
    for i in range(config.sub_populations):
        queues.append(mp.Queue())
    q = mp.Queue()
    data_queue = mp.Queue()

    for i in range(config.sub_populations):
        id_out = (config.sub_populations - 1) if i == 0 else i - 1
        processes.append(
            mp.Process(target=genetic_algorithm, args=(pop, knn, config, i, queues[i], queues[id_out], q, data_queue)))

    for p in processes:
        p.start()

    for i in range(config.sub_populations):
        final_pop += q.get()
        gen_and_eval.append(data_queue.get())

    for p in processes:
        p.terminate()

    final_pop = tools.sortNondominated(final_pop, config.individuals, first_front_only=True)[0]
    data_backup = {'folder_dataset': config.folder_dataset, 'generations_convergence': config.generations_convergence,
                   'individuals': config.individuals,
                   'percentage_fs': config.percentage_fs, 'maximum_generations': config.max_generations,
                   'subpopulations': config.sub_populations, 'migrations': config.migrations,
                   'period': config.period, 'grain': config.grain,
                   'evaluation_version': config.evaluation_version, 'accuracy_convergence': config.accuracy_convergence,
                   'sd_convergence': config.sd_convergence, 'k': config.k, 'experiment_name': config.experiment_name}

    print("************ PARETO FRONT ************")
    best_individuals = []
    features = [g for g in final_pop[0]]
    features_selected = [features]
    ac, kappa = knn.calculate_accuracy_test(final_pop[0])
    final_pop[0].test_kappa = kappa
    final_pop[0].test_accuaracy = ac
    final_pop[0].features = features
    best_individuals.append(final_pop[0])

    for i in range(len(final_pop)):
        features = [g for g in final_pop[i]]
        try:
            features_selected.index(features)
        except:
            ac, kappa = knn.calculate_accuracy_test(final_pop[i])
            print("Accuracy Test:", ac, "** Kappa coefficient:", kappa, "** fitness: ", final_pop[i].fitness.values)
            final_pop[i].test_kappa = kappa
            final_pop[i].test_accuaracy = ac
            final_pop[i].features = features
            best_individuals.append(final_pop[i])
            features_selected.append(features)
    if config.evaluation_version != 1:
        os.remove(config.project_path + "/db/aux/data_to_c.npy")

    experiment = {'pareto_front': best_individuals, 'gens_and_evals': gen_and_eval}
    data_backup['experiment'] = experiment

    return data_backup
