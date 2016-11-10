from repository import Repository
from configuration import config

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors, datasets
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB

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

        #Anzahl an Nachbarn die verwendet werden.
        n_neighbors = 15

        algorithm = 'auto'
        #algorithm = 'ball_tree'
        #algorithm = 'kd_tree'
        #algorithm = 'brute'

        #Mit unterschiedlichen Gewichtungen klassifizieren
        #weight function used in prediction. Possible values:
        #‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        #‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
        for weights in ['uniform', 'distance']:

            #Classifier erzeugen
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm)

            #training
            clf.fit(self.X_train, [self.repository.locations.keys().index(tuple(l)) for
                                 l in self.y_train])

            #testing
            predict = clf.score(self.X_test, [self.repository.locations.keys().index(tuple(l)) for
                                  l in self.y_test])

            print predict


    def classifyBayesGausch(self):

        #The number of mixture components. Depending on the data and the value of the weight_concentration_prior the model can decide to not use all the components by setting some component weights_ to values very close to zero. The number of effective components is therefore smaller than n_components.
        n_components = 1

        covariance_type = 'full'
        #covariance_type = 'tied'
        #covariance_type = 'diag'
        #covariance_type = 'spherical'

        #The number of initializations to perform. The result with the highest lower bound value on the likelihood is kept.
        n_init = 1

        #The method used to initialize the weights, the means and the covariances
        init_params = 'kmeans'
        #init_params = 'random'

        clf = BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_type, n_init=n_init, init_params=init_params)

        #training
        clf.fit(self.X_train, [self.repository.locations.keys().index(tuple(l)) for
                             l in self.y_train])

        #testing
        predict = clf.score(self.X_test, [self.repository.locations.keys().index(tuple(l)) for
                              l in self.y_test])

        print predict

    def classifyNaiveBayes(self):

        #classifier
        clf = GaussianNB()

        weight = np.full((len(self.X_train), 1), 10, dtype=np.int)

        #training
        clf.fit(self.X_train, [self.repository.locations.keys().index(tuple(l)) for
                             l in self.y_train], weight)

        #testing
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