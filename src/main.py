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

import sys

from time import time

from knn import Knn
from wrapper import feature_selection
from database_functions import save_experiment
from config import Config

__author__ = 'Juan Carlos Gómez-López'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.0'
__maintainer__ = 'Juan Carlos Gómez-López'
__email__ = 'goloj@ugr.es'
__status__ = 'Development'


def main():
    """
    Main function focused on executing the entire wrapper and storing the data on the MongoDB database.
    """
    # Config object
    config = Config()

    start_time = time()

    knn = Knn(config=config)

    # Calling to the genetic algorithm
    data_backup = feature_selection(knn=knn, config=config)

    end_time = time()
    print("Execution time: ", end_time - start_time)
    data_backup['experiment']['execution_time'] = end_time - start_time

    # Saving data to the database
    save_experiment(data_backup=data_backup)
    print(f'Generations: {config.generations_convergence} // Population: {config.individuals} DONE')

    sys.exit(0)


if __name__ == "__main__":
    main()
