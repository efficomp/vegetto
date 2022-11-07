"""
    @file knn.py

    @brief This file contains all the functions related to the kmeans algorithm.

    @details The fitness of an individual is calculated through the BCSS and WCSS values. This files provides the
    functions to calculate them.

    All the information about the different methods is explained in each function.

    This work has been funded by the Spanish Ministry of Science, Innovation, and Universities under grant
    PGC2018-098813-B-C31 and ERDF funds

    The Python version used is 3.6.

    This software make use of external libraries such as:

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

import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn import preprocessing

from src.config import Config


class Knn():
    def __init__(self, config: Config):
        """
        Class Constructor
        :param filename: EEG file name
        :param number_of_features: Number of features to select from file
        """

        self.data_train = np.load(r"db/UCI/Robust/{}/data_train.npy".format(config.folder_dataset),
                                  allow_pickle=True)
        self.labels_train = np.load("db/UCI/Robust/{}/labels_train.npy".format(config.folder_dataset),
                                    allow_pickle=True).astype('int')
        self.data_test = np.load("db/UCI/Robust/{}/data_test.npy".format(config.folder_dataset),
                                 allow_pickle=True)
        self.labels_test = np.load("db/UCI/Robust/{}/labels_test.npy".format(config.folder_dataset),
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

    def calculate_accuracy_validation(self, individual):
        """
        Function to calculate the centroids and the WCSS of an individual

        :param individual: Individual of the population

        :param max_iter_kmeans: maximum number of iterations performed by the  Kmeans
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
        data_to_knn_train = self.data_train[:, individual]
        data_to_knn_test = self.data_test[:, individual]

        if self.k == -1:
            model = KNeighborsClassifier(n_neighbors=int(round(math.sqrt(len(data_to_knn_train)))), algorithm='brute')
        else:
            model = KNeighborsClassifier(n_neighbors=self.k, algorithm='brute')

        model.fit(data_to_knn_train, self.labels_train)

        return accuracy_score(model.predict(data_to_knn_test), self.labels_test), cohen_kappa_score(
            model.predict(data_to_knn_test), self.labels_test)
