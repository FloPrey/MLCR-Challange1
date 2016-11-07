from repository import Repository
from configuration import config

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV

class Analyzer(object):
    def __init__(self):
        self.repository = Repository(config)
        self.dataset, self.labels = self.repository.get_dataset_and_labels()

    def learning(self):
        # Ensure that there are no NaNs
        self.dataset.fillna(-85)
        # Split the dataset into training (90 \%) and testing (10 \%)
        X_train, X_test, self.y_train, y_test = train_test_split(self.dataset, self.labels,
                                                            test_size=0.1)

        cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2,
                          random_state=0)

        # Define the classifier to use
        estimator = SVC(kernel='linear')

        # Define parameter space.
        gammas = np.logspace(-6, -1, 10)

        # Use Test dataset and use cross validation to find bet hyper-parameters.
        classifier = GridSearchCV(estimator=estimator, cv=cv,
                                  param_grid=dict(gamma=gammas))
        classifier.fit(X_train, [self.repository.locations.keys().index(tuple(l)) for
                                 l in self.y_train])

        # Test final results with the testing dataset
        classifier.score(X_test, [self.repository.locations.keys().index(tuple(l)) for
                                  l in y_test])

    def convert(self):
        # Convert to numerical labels
        numerical_labels = [self.repository.locations.keys().index(tuple(l)) for
                            l in self.y_train]

        # Convert back to coordinate labels
        coordinate_labels = [repository.locations.keys()[i] for i in c_output]