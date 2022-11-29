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

import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn import preprocessing

from config import Config

__author__ = 'Juan Carlos Gómez-López'
__copyright__ = 'Copyright 2022, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.0'
__maintainer__ = 'Juan Carlos Gómez-López'
__email__ = 'goloj@ugr.es'
__status__ = 'Development'


class Knn():
    def __init__(self, config: Config):
        """
        Constructor.

        :param config: Config object where all the hyperparameter values are loaded
        :type Config: :py:mod:`config`

        """

        self.data_train = np.load(r"db/{}/data_train.npy".format(config.folder_dataset),
                                  allow_pickle=True)
        self.labels_train = np.load("db/{}/labels_train.npy".format(config.folder_dataset),
                                    allow_pickle=True).astype('int')
        self.data_test = np.load("db/{}/data_test.npy".format(config.folder_dataset),
                                 allow_pickle=True)
        self.labels_test = np.load("db/{}/labels_test.npy".format(config.folder_dataset),
                                   allow_pickle=True).astype('int')

        le = preprocessing.LabelEncoder()
        le.fit(self.labels_train)
        self.labels = le.transform(self.labels_train)

        le = preprocessing.LabelEncoder()
        le.fit(self.labels_test)
        self.labels_test = le.transform(self.labels_test)

        self.k = config.k
        self.accuracy_validation = 0.0
        self.number_of_selected_features = 0.0

    def calculate_kappa_coefficiente_validation(self, individual):
        """
        Calculation of the validation Kappa coefficient.

        :param individual: Chromosome of the individual (selected features)
        :type Individual: Individual

        """

        data_to_knn = self.data_train[:, individual]

        data_train, data_validation, labels_train, labels_validation = train_test_split(data_to_knn, self.labels_train,
                                                                                        test_size=0.5,
                                                                                        stratify=self.labels_train)
        if self.k == -1:
            model = KNeighborsClassifier(n_neighbors=int(round(math.sqrt(len(data_train)))), algorithm='brute')
        else:
            model = KNeighborsClassifier(n_neighbors=self.k, algorithm='brute')

        model.fit(data_train, labels_train)

        self.accuracy_validation = cohen_kappa_score(model.predict(data_validation), labels_validation)
        self.number_of_selected_features = len(individual)

    def calculate_accuracy_test(self, individual):
        """
        Calculation of the test accuracy.

        :param individual: Chromosome of the individual (selected features)
        :type Individual: Individual

        """
        data_to_knn_train = self.data_train[:, individual]
        data_to_knn_test = self.data_test[:, individual]

        if self.k == -1:
            model = KNeighborsClassifier(n_neighbors=int(round(math.sqrt(len(data_to_knn_train)))), algorithm='brute')
        else:
            model = KNeighborsClassifier(n_neighbors=self.k, algorithm='brute')

        model.fit(data_to_knn_train, self.labels_train)

        return accuracy_score(model.predict(data_to_knn_test), self.labels_test), cohen_kappa_score(
            model.predict(data_to_knn_test), self.labels_test)
