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

import xml.etree.ElementTree as ET
import os
import sys

__author__ = 'Juan Carlos Gómez-López'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.0'
__maintainer__ = 'Juan Carlos Gómez-López'
__email__ = 'goloj@ugr.es'
__status__ = 'Development'


class Config:
    """Define the config class for setting all the hyperparameter values"""

    def __init__(self):
        """Create an empty config object."""
        tree = ET.parse(r'{}/config.xml'.format(os.getcwd()))
        root = tree.getroot()

        self.folder_dataset = root[0].text
        self.features = int(root[1].text)
        self.executions = int(root[2].text)
        self.individuals = int(root[3].text)
        self.generations_convergence = int(root[4].text)
        self.max_generations = int(root[5].text)
        self.sub_populations = int(root[6].text)
        self.migrations = int(root[7].text)
        self.evaluation_version = int(root[8].text)
        self.processes = int(root[9].text)
        self.fitness_evolution = int(root[10].text)
        self.percentage_fs = float(root[11].text)
        self.accuracy_convergence = float(root[12].text)
        self.sd_convergence = float(root[13].text)
        self.k = int(root[14].text)
        self.paper = root[15].text
        self.node = root[16].text
        self.pc = float(root[17].text)
        self.pm = float(root[18].text)
        self.pmr = float(root[19].text)
        self.generations_migration = int(root[20].text)
        self.decision_features = float(root[21].text)
        self.project_path = root[22].text

        if self.sub_populations < 0:
            print("Error: The number of sub-populations are not correct (the minimum has to be 1)")
            sys.exit(-1)

        if self.migrations < 0:
            print("Error: The number of migrations must be greater than 0")
            sys.exit(-1)

        if self.sub_populations == 1 and not self.migrations == 0:
            print("Error: If there is one subpopulations only, the number of migrations must be 0")
            sys.exit(-1)

        if self.evaluation_version < 1 or self.evaluation_version > 6:
            print("Error: The evaluation type is not correct")
            sys.exit(-1)

        if self.fitness_evolution == 1 and self.sub_populations > 1:
            print("Error: If sub_populations is greater than 1 the fitness evolution analysis can't be performed")
            sys.exit(-1)

        if self.percentage_fs > 1 or self.percentage_fs < 0.001:
            print("Error: Wrong percentage of selected features. It has to be between 0.001-0.999")
            sys.exit(-1)
