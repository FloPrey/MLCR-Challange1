from repository import Repository
from configuration import config

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors, datasets

class Analyzer(object):
    def loadData(self):
        self.repository = Repository(config)
        self.dataset, self.labels = self.repository.get_dataset_and_labels()


    def improveData(self):
        # Ensure that there are no NaNs
        self.dataset = self.dataset.fillna(-85)
        # Split the dataset into training (90 \%) and testing (10 \%)

    def splitData(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.labels,
                                                                                test_size=0.1)

        print(self.X_train)
        print(self.y_train)

        self.cv = ShuffleSplit(self.X_train.shape[0], n_iter=10, test_size=0.2,
                               random_state=0)

    def classifyKNearest(self):
        n_neighbors = 15

        for weights in ['uniform', 'distance']:

            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(self.X_train, [self.repository.locations.keys().index(tuple(l)) for
                                 l in self.y_train])
            predict = clf.score(self.X_test, [self.repository.locations.keys().index(tuple(l)) for
                                  l in self.y_test])

            print predict

    def classifySVC(self):
        # Define the classifier to use
        estimator = SVC(kernel='linear')

        # Define parameter space.
        gammas = np.logspace(-6, -1, 10)

        # Use Test dataset and use cross validation to find bet hyper-parameters.
        classifier = GridSearchCV(estimator=estimator, cv=self.cv,
                                  param_grid=dict(gamma=gammas))
        classifier.fit(self.X_train, [self.repository.locations.keys().index(tuple(l)) for
                                 l in self.y_train])

        # Test final results with the testing dataset
        predict = classifier.score(self.X_test, [self.repository.locations.keys().index(tuple(l)) for
                                  l in self.y_test])

        print predict


    def convert(self):
        # Convert to numerical labels
        numerical_labels = [self.repository.locations.keys().index(tuple(l)) for
                            l in self.y_train]

        print(numerical_labels)

        # Convert back to coordinate labels
        coordinate_labels = [self.repository.locations.keys()[i] for i in numerical_labels]
        print(coordinate_labels)