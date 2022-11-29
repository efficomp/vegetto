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

from pymongo import MongoClient


def save_experiment(data_backup):
    """
    Here, a wrapper experiment is stored in the database

    :param data_backup: Dictionary with all the data
    :type dict: :py:class:`dict`

    """
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
                               'k': data_backup['k'], 'experiment_name': data_backup['experiment_name']})
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
             'sd_convergence': data_backup['sd_convergence'], 'k': data_backup['k'],
             'experiment_name': data_backup['experiment_name']},
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
             'sd_convergence': data_backup['sd_convergence'], 'k': data_backup['k'],
             'experiment_name': data_backup['experiment_name'],
             'experiment': [data_backup['experiment']]})
