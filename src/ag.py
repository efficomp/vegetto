"""
    @file ag.py

    @brief This file contains all the necessary functions to execute the feature selection method.

    @details The feature selection method is solved by a wrapper, using an evolutionary multi-objective optimization
    procedure along with a clustering algorithm, specially NSGA-II and K-means respectively.

    All the information about the different methods is explained in each function.

    This work has been funded by the Spanish Ministry of Science, Innovation, and Universities under grant
    PGC2018-098813-B-C31 and ERDF funds.

    This software makes use of some built-in modules such as random, math, time, functools and mulitprocessing. The
    Python version used is 3.6.

    These modules aside, this software make use of external libraries such as:

        -Deap: created by François-Michel De Rainville, Félix-Antoine Fortin and Christian Gagné (Copyright, 2012) and
        released under GNU Lesser general public license. The Github Deap repository can be found in:
        https://github.com/DEAP/deap.

        -Sklearn: created by David Cournapeau (Copyright, 2007-2020) and released under BSD 3-Clause License.The Github
        Sklearn repository can be found in: https://github.com/scikit-learn/scikit-learn.

        -Pandas: created by Wes McKinney as Benevolent Dictator for Life and the Pandas's Team (Copyright 2008-2011,
        AQR Capital Management and 2011-2020, Open source contributors) and released under BSD 3-Clause License. The
        Github Pandas repository can be found in: https://github.com/pandas-dev/pandas.

        -Numpy: created by Travis Oliphant (Copyright, 2005-2020) and released under BSD 3-Clause License. The Github
        Numpy repository can be found in: https://github.com/numpy/numpy.

    @author Juan Carlos Gómez López

    @date 30/04/2020

    @version 1.0

    @copyright Licensed under GNU GPL-3.0-or-later

    «Copyright 2020 EffiComp@ugr.es»
"""
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

from src.knn import Knn
from src.config import Config
from knn_library import *

toolbox = base.Toolbox()


def fill_chromosome(p, features):
    """
    Function to fill the chromosome.

    :param p: p
    :param features: features

    :return feature value
    """
    aux_features = []

    for i in range(features):
        if random.uniform(0, 1) < p:
            aux_features.append(i)

    return np.array(aux_features)


def random_gen(features):
    return np.array([random.randint(0, features - 1)])


def recombination_and_mutation(pool, config: Config):
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


def mutate(individual, features, pm):
    all_features = np.arange(features)
    not_selected = np.setdiff1d(all_features, individual)

    to_be_kept = individual[np.random.random(len(individual)) > pm]

    to_be_added = not_selected[np.random.random(not_selected.size) <= pm]

    return np.sort(np.concatenate((to_be_kept, to_be_added)))


def evaluate(ind, knn: Knn):
    """
    Function to evaluate the population.

    :param individual: individual of the population
    :param knn: knn object
    :param config: Config object

    :return BCSS and WCSS
    """
    knn.calculate_accuracy_validation(ind)
    ind.fitness.values = (knn.accuracy_validation, knn.number_of_selected_features)
    return ind


def migration(pop, subpops, sub_population_size):
    """
    Function to performance the migration between different sub-populations: every subpopulation gives the half of its
    Pareto front to the next subpopulation following a ring topology.

    :param pop: population
    :param subpops: number of subpopulations
    :param sub_population_size: subpopulation size

    :return None
    """

    for sub_pop in range(subpops):
        pareto_front = tools.sortNondominated(pop[sub_pop], sub_population_size, first_front_only=True)
        if len(pareto_front[0]) > 1:
            dest_sub_pop = 0 if sub_pop == (subpops - 1) else sub_pop + 1
            pop[dest_sub_pop] += pop[sub_pop][0:math.ceil(len(pareto_front[0]) / 2)]


def evaluation(population, knn: Knn, config: Config):
    for i in range(len(population)):
        if len(population[i]) == 0:
            population[i] = toolbox.population_for_empty_individual(n=1)[0]
    if config.evaluation_version == 1:
        population = [toolbox.evaluate(ind, knn) for ind in population]
    else:
        pop_for_library = Population([Individual(ind, ind.fitness.values) for ind in population])
        config_for_library = Configuration(config.features, config.sub_populations, config.evaluation_version, config.k,
                                           config.project_path + "/db/aux/data_to_c.npy",
                                           len(np.unique(knn.labels_train)), config.decision_features)

        fitness = pop_for_library.evaluate(config_for_library)

        for index in range(len(population)):
            population[index].fitness.values = [cohen_kappa_score(fitness[index * 2], fitness[(index * 2) + 1]),
                                                len(population[index])]


def genetic_algorithm(knn: Knn, config: Config, pop, sub_pop, q_in, q_out, q, data_queue):
    tr, te, ts = 0.0, 0.0, 0.0
    m, gen = 0, 0
    finish = False
    tt = time()
    evaluation(pop[sub_pop], knn, config)
    te += (time() - tt)

    tt = time()
    pop[sub_pop] = toolbox.selectNSGA2(pop[sub_pop], len(pop[sub_pop]))
    ts += (time() - tt)
    mean_accuracy_convergence, sd_convergence = np.array([]), np.array([])

    accuracy_evolution, features_evolution = np.array([]), np.array([])

    number_of_evaluations = len(pop[sub_pop])

    while not finish:
        # -------- Starting the evaluation process --------

        if not q_out.empty():
            pareto_to_pop = q_out.get_nowait()
            pop[sub_pop] += pareto_to_pop
            pop[sub_pop] = toolbox.selectNSGA2(pop[sub_pop], config.individuals)

        # -------- Creation of the pool --------
        tt = time()
        pool = tools.selTournamentDCD(pop[sub_pop], config.individuals)
        offspring = recombination_and_mutation(pool, config)
        tr += (time() - tt)

        # ---------------------------------------------- EVALUATION -----------------------------------------------
        tt = time()
        evaluation(offspring, knn, config)
        te += (time() - tt)
        number_of_evaluations += len(offspring)

        # ----------------------------------------------- SELECTION -----------------------------------------------
        tt = time()
        pop[sub_pop] = toolbox.selectNSGA2(offspring + pop[sub_pop], config.individuals)
        ts += (time() - tt)

        data_ac = [x.fitness.values[0] for x in pop[sub_pop]]
        mean_accuracy_convergence = np.append(mean_accuracy_convergence, st.mean(data_ac))
        sd_convergence = np.append(sd_convergence, st.pstdev(data_ac))
        print("Max: ", max(data_ac), "Mean: ", st.mean(data_ac), " SD: ", st.pstdev(data_ac))

        if config.fitness_evolution:
            data_f = [x.fitness.values[1] for x in pop[sub_pop]]
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
            pareto_front = tools.sortNondominated(pop[sub_pop], config.individuals, first_front_only=True)[0]
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

    q.put(pop[sub_pop])
    if config.fitness_evolution:
        data_queue.put([number_of_evaluations, gen, te, ts, list(accuracy_evolution), list(features_evolution)])
    else:
        data_queue.put([number_of_evaluations, gen, te, ts])


def features_selection(config: Config, knn: Knn):
    """
        Function to perform the evolutionary procedure

        :param config: Config object
        :param knn: Knn object

        :return: hypervolume and tf (total features selected)
    """

    # -----------------
    # Defining genetics operators
    # -----------------

    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    toolbox.register("fill_chromosome", fill_chromosome)

    chromosome_generator = partial(fill_chromosome, config.percentage_fs, config.features)
    empty_chromosome = partial(random_gen, config.features)
    toolbox.register("individual", tools.initIterate, creator.Individual, chromosome_generator)
    toolbox.register("fill_empty_individual", tools.initIterate, creator.Individual, empty_chromosome)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("population_for_empty_individual", tools.initRepeat, list, toolbox.fill_empty_individual)

    toolbox.register("evaluate", evaluate)

    toolbox.register("crossover", tools.cxUniform, indpb=0.5)

    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    toolbox.register("selectNSGA2", tools.selNSGA2)
    toolbox.register("selectTournament", tools.selTournament, tournsize=2)

    """
    -----------------
    Genetic Algorithm
    -----------------
    """

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
            mp.Process(target=genetic_algorithm, args=(knn, config, pop, i, queues[i], queues[id_out], q, data_queue)))

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
                   'sd_convergence': config.sd_convergence, 'k': config.k, 'paper': config.experiment_name}

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
