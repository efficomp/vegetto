"""
    @file database_functions.py

    @brief This file contains all the necessary functions to manage the database.

    @details All the data related to the evolutionary procedure (times, hypervolume, generations, among others) are
    saved and updated on the database in order to generate the plots.

    All the information about the different methods is explained in each function.

    This work has been funded by the Spanish Ministry of Science, Innovation, and Universities under grant
    PGC2018-098813-B-C31 and ERDF funds

    The Python version used is 3.6.

    This software make use of external libraries such as:

        -Pymongo: created by Mike Dirolf (Copyright, 2008-present) and released under Apache license. The Github Pymongo
        repository can be found in: https://github.com/mher/pymongo.

    @author Juan Carlos Gómez López

    @date 30/04/2020

    @version 1.0

    @copyright Licensed under GNU GPL-3.0-or-later

    «Copyright 2020 EffiComp@ugr.es»
"""
import sys

from pymongo import MongoClient


def save_experiment(data_backup):
    # Initialize database client
    client = MongoClient()
    db = client.Vegetto

    results = db.results

    result = results.find_one({'dataset': data_backup['folder_dataset'],
                               'generations_convergence': data_backup['generations_convergence'],
                               'individuals': data_backup['individuals'],
                               'percentage_fs': data_backup['percentage_fs'],
                               'maximum_generations': data_backup['maximum_generations'],
                               'subpopulations': data_backup['subpopulations'], 'migrations': data_backup['migrations'],
                               'evaluation_version': data_backup['evaluation_version'],
                               'grain': data_backup['grain'], 'period': data_backup['period'],
                               'accuracy_convergence': data_backup['accuracy_convergence'],
                               'sd_convergence': data_backup['sd_convergence'],
                               'k': data_backup['k'], 'paper': data_backup['paper']})
    pareto_front = []
    for i in range(len(data_backup['experiment']['pareto_front'])):
        pareto_front.append({'test_accuracy': data_backup['experiment']['pareto_front'][i].test_accuaracy,
                             'test_kappa': data_backup['experiment']['pareto_front'][i].test_kappa,
                             'validation_kappa': data_backup['experiment']['pareto_front'][i].fitness.values[0],
                             'features': [int(x) for x in data_backup['experiment']['pareto_front'][i].features]})
    data_backup['experiment']['pareto_front'] = pareto_front

    if result:
        experiment_aux = result.get('experiment')
        experiment_aux.append(data_backup['experiment'])
        db.results.update_one(
            {'dataset': data_backup['folder_dataset'],
             'generations_convergence': data_backup['generations_convergence'],
             'individuals': data_backup['individuals'],
             'percentage_fs': data_backup['percentage_fs'], 'maximum_generations': data_backup['maximum_generations'],
             'subpopulations': data_backup['subpopulations'], 'migrations': data_backup['migrations'],
             'evaluation_version': data_backup['evaluation_version'],
             'grain': data_backup['grain'], 'period': data_backup['period'],
             'accuracy_convergence': data_backup['accuracy_convergence'],
             'sd_convergence': data_backup['sd_convergence'], 'k': data_backup['k'], 'paper': data_backup['paper']},
            {'$set': {'experiment': experiment_aux}})

    else:
        results.insert_one(
            {'dataset': data_backup['folder_dataset'],
             'generations_convergence': data_backup['generations_convergence'],
             'individuals': data_backup['individuals'],
             'percentage_fs': data_backup['percentage_fs'], 'maximum_generations': data_backup['maximum_generations'],
             'subpopulations': data_backup['subpopulations'], 'migrations': data_backup['migrations'],
             'evaluation_version': data_backup['evaluation_version'],
             'grain': data_backup['grain'], 'period': data_backup['period'],
             'accuracy_convergence': data_backup['accuracy_convergence'],
             'sd_convergence': data_backup['sd_convergence'], 'k': data_backup['k'], 'paper': data_backup['paper'],
             'experiment': [data_backup['experiment']]})

