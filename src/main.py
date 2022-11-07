"""
    @file main.py

    @brief Main file.

    @details This file calls all the necessary functions to perform the evolutionary procedure and create the plots.

    All the information about the different methods is explained in each function.

    This work has been funded by the Spanish Ministry of Science, Innovation, and Universities under grant
    PGC2018-098813-B-C31 and ERDF funds.

    This software makes use of some built-in modules such as time and sys. The Python version used is 3.6.

    @author Juan Carlos Gómez López

    @date 30/04/2020

    @version 1.0

    @copyright Licensed under GNU GPL-3.0-or-later

    «Copyright 2020 EffiComp@ugr.es»
"""

import sys

from time import time

from src.knn import Knn
from src.ag import features_selection
from src.database_functions import save_experiment
from src.config import Config


def main():
    """
    Main function
    """
    # Config object
    config = Config()

    start_time = time()

    knn = Knn(config=config)

    # Calling to the genetic algorithm
    data_backup = features_selection(config=config, knn=knn)

    end_time = time()
    print(end_time - start_time)

    # Saving data to the database
    save_experiment(data_backup=data_backup)
    print(f'Generations: {config.generations_convergence} // Population: {config.individuals} DONE')

    sys.exit(0)


if __name__ == "__main__":
    main()
