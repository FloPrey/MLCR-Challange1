from repository import Repository
from configuration import config

import numpy as np
import pandas as pandas
import os.path
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors, datasets
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import math

class Analyzer(object):
    
    
    """Method that loads the datapoints from the repository and transfers them into lists and
    numpy arrays."""
    def loadData(self):
        
        self.repository = Repository(config)
        self.dataset, self.labels = self.repository.get_dataset_and_labels()
        
        with open("locations.txt") as inputFile:
            locationList = [tuple(line.split(',')) for line in inputFile.readlines()]
    
        # create lists containing the floor and coordinate values
        floors = [floor[2].strip() for floor in locationList]
        coordinates = [(float(floor[0]), float(floor[1])) for floor in locationList]
    
        # Convert coordinate values to numerical labels
        self.numerical_labels = np.array([self.repository.locations.keys().index(tuple(l)) for
                            l in self.labels])
        
        # Y-Value for the learning algorithm --> Floor Labels
        floorList = []
         
        # Fill floor labels in correct order according to the coordinate labels
        for nlabel in self.numerical_labels:
            floorIndex = coordinates.index(self.repository.locations.keys()[nlabel])
            floorList.append(floors[floorIndex])
        
        self.floorLabels = np.array(floorList)
        
        self.numericalFloorLabels = self.createNumericalFloorLabels(floorList)
        
    """Helper Method to change the String values of the floor string labels into numerical values."""    
    def createNumericalFloorLabels(self, floorList):
        
        floorSet = list(set(floorList))
        numericalSet = range(0, len(floorList)-1)
        
        for i in range(len(floorList)):
            for j in range(len(floorSet)):
                if floorList[i] == floorSet[j]:
                    floorList[i] = numericalSet[j]
                    break
                
        return floorList

    def diff(self, a, b):
        b = set(b)
        return [aa for aa in a if aa not in b]

    def improveData(self, percentage_value, normalize):

        # ############################ STEP 0 #############################
        # if an optimized-dataset.csv is found, do not optimize it again.
        if os.path.isfile("optimized-dataset.csv"):
            print ("optimised Dataset found. Create new one? [y/n]")
            input = raw_input()
            if input == "n":
                print "loading existing optimised Dataset"
                self.dataset = pandas.read_csv("optimized-dataset.csv", sep='\t')
                print "dataset size: ", self.dataset.shape
                return

        deletecolumns = []
        print "Size of Dataset before Selection: ", self.dataset.shape

        # ############################## STEP 1 ############################
        # Fill NaN with -85
        print "Filling NaNs"
        self.dataset = self.dataset.fillna(-85)

        # ############################# STEP 2 #############################
        # Values smaller then -85db (=unreachable) are replaced by -85db
        if (normalize == True):
            print "Normalizing unreachable Access points to -85dB"
            self.dataset[self.dataset < -85] = -85

        # ############################ STEP 3 #############################
        # add the location labels
        numerical_labels = np.array([self.repository.locations.keys().index(tuple(l)) for l in self.labels])
        numerical_labels = map(int, numerical_labels)
        self.dataset['location',] = numerical_labels

        # ############################ STEP 4 #############################
        # clean all AP with a maximum value of -85db
        # this will increase the speed of the next steps
        print "removing Access points that are never reachable"
        for column in self.dataset:
            datalist = self.dataset[column].tolist()
            if max(datalist) <= -85:
                deletecolumns.append(column)

        self.dataset = self.dataset.drop(deletecolumns, axis=1)


        # ############################ STEP 5 #############################
        # remove AP which do not appear more often then *percentage_value* per Location
        print "removing Access Points that only appear less then: ", percentage_value, "% per location"

        if percentage_value > 0:
            goodAP = []
            badAP = []

            # count amount of different locations we have
            loc_amount = len(set(numerical_labels))

            for location in range(0, loc_amount):

                # check only Data for current location
                location_dataset = self.dataset.loc[lambda dataset: dataset.location == location, :]
                print "location: ", location, "size: ", location_dataset.shape, "GoodAP: ", len(goodAP)

                # amount of measurement data for current location
                measure_amount = len(self.dataset.loc[lambda dataset: dataset.location == location, :])

                # if a AP appears >70% in the data for a location its a good access point.
                threshhold = math.ceil(measure_amount * percentage_value/100)

                for column in location_dataset:
                    if column != "location" and column not in goodAP:
                        datalist = location_dataset[column].tolist()
                        amount = sum(i > -85 for i in datalist)

                        if amount > threshhold:
                            goodAP.append(column)
                        else:
                            badAP.append(column)

            badAP = set(badAP)
            goodAP = set(goodAP)

            deletecolumns = self.diff(badAP, goodAP)

            # delete those columns
            self.dataset = self.dataset.drop(deletecolumns, axis = 1)

            #delete the location column again
            self.dataset = self.dataset.drop('location', axis = 1)

        print "Size of Dataset after Selection: ", self.dataset.shape

        # ############################ STEP 6 #############################
        # save cleaned dataset as csv (for further use)

        self.dataset.to_csv("optimized-dataset.csv", sep='\t')



    def splitData(self, x_values, y_values):
        
        X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.1)


        self.cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2,
                               random_state=0)
        
        return X_train, X_test, y_train, y_test



    """Method to train a classifier to learn to predict the floor of the given datapoints."""
    def predictFloor(self):
        
        X_train, X_test, y_train, y_test = self.splitData(self.dataset, self.floorLabels)
        
        # save the testset globally to use again later for location prediction
        self.X_dataset = X_test
        self.y_dataset = y_test
        
        print "Starting training process for the Floor Prediction!"
        print "------------------------------------"
        
        classifier = self.classifyRandomForest(X_train, X_test, y_train, y_test)
        
        self.floorPrediction = classifier.predict(self.X_dataset)
        
        print "Floors that have been predicted:"
        print self.floorPrediction
        print "------------------------------------"
        
        
    """This method predicts the location of the previously loaded dataset. 
    The dataset must also contain the floorLabels to increase accurracy. 
    The original labels are added for the training process, however for the
    prediction we use the previously predicted floors from the predictFloor method."""    
    def predictLocation(self):
        
        # add true floor labels to the dataset for training
        self.dataset['floorLabels'] = self.numericalFloorLabels
        
        # split dataset into train and test sets
        X_train, X_test, y_train, y_test = self.splitData(self.dataset, self.numerical_labels)
        
        print "Starting training process for the Location Prediction!"
        print "------------------------------------"
        
        # train a classifier with the training and test data
        classifier = self.classifyRandomForest(X_train, X_test, y_train, y_test)
        
        # add predicted floor labels to the dataset
        self.X_dataset['floorLabels'] = self.createNumericalFloorLabels(self.floorPrediction)
        
        # predict the locations usind the dataset, but with predicted labels instead
        self.locationPrediction = classifier.predict(self.X_dataset)
        
        print "Locations that have been predicted:"
        print self.locationPrediction
        print "------------------------------------"
        
        self.errorValue = self.calculateAverageLocationError()
        
        print "Average - error value of missclassified locations in meter:"
        print self.errorValue
        print "------------------------------------"
        
    """Method to evaluate the results of the incorrect predicted locations. It takes the
    latitude and longitude representation of the location labels and calculates the difference 
    between the predicted and the ground truth values in meter."""    
    def calculateAverageLocationError(self):
        
        # transform the predicted location labels back to coordinates
        predictedCoordinate_labels = [self.repository.locations.keys()[i] for i in self.locationPrediction]
        
        # transform the numpy array into a list of tupels
        labelList = tuple(map(tuple, self.labels))
        
        # create a list of tupels containing all the missclassified locations and their actual coordinates
        errorList = []
        for predicted, truth in zip(predictedCoordinate_labels, labelList):
            if predicted != truth:
                errorList.append((predicted,truth))
                
        print "Percentage of incorrect location predictions:"
        print float(float(len(errorList))/float(len(labelList)))
        print "------------------------------------"
        
        
        # calculate the error value in meter between the incorrect predicted values and the ground truth
        error = 0.0
        for item in errorList:
            error = error + self.calculateError(item[0], labelList[1])
        
        return float(error/float(len(errorList)))
    
    
    def classifyRandomForest(self, X_train, X_test, y_train, y_test):   
        
        estimators = 200
        
        clf = RandomForestClassifier(n_estimators=estimators)
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) 
        
        print "RandomForest-Classifier trained - with score: "
        print score
        print "------------------------------------"
        
        return clf

    def classifyKNearest(self, X_train, X_test, y_train, y_test):

        #Anzahl an Nachbarn die verwendet werden.
        n_neighbors = 15

        algorithm = 'auto'
        #algorithm = 'ball_tree'
        #algorithm = 'kd_tree'
        #algorithm = 'brute'

        #Mit unterschiedlichen Gewichtungen klassifizieren
        #weight function used in prediction. Possible values:
        #uniform : uniform weights. All points in each neighborhood are weighted equally.
        #distance : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
        for weights in ['uniform', 'distance']:

            #Classifier erzeugen
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm)

            #training
            clf.fit(X_train, y_train)

            #testing
            score = clf.score(X_test, y_test)

            print "KNN-Classifier trained - with score: "
            print score
            print "------------------------------------"
            
            return clf


    def classifyBayesGausch(self, X_train, X_test, y_train, y_test):

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
        clf.fit(X_train, y_train)

        #testing
        score = clf.score(X_test, y_test)

        print "BayesGausch-Classifier trained - with score: "
        print score
        print "------------------------------------"
        return clf

    def classifyNaiveBayes(self, X_train, X_test, y_train, y_test):

        #classifier
        clf = GaussianNB()

        weight = np.full((len(X_train), 1), 10, dtype=np.int)

        #training
        clf.fit(X_train, y_train, weight)

        #testing
        score = clf.score(X_test, y_test)

        
        print "NaiveBayes-Classifier trained - with score: "
        print score
        print "------------------------------------"
        return clf

    def classifySVC(self, X_train, X_test, y_train, y_test):
        # Define the classifier to use
        estimator = SVC(kernel='linear')

        # Define parameter space.
        gammas = np.logspace(-6, -1, 10)

        # Use Test dataset and use cross validation to find bet hyper-parameters.
        classifier = GridSearchCV(estimator=estimator, cv=self.cv,
                                  param_grid=dict(gamma=gammas))
        classifier.fit(X_train, y_train)

        # Test final results with the testing dataset
        score = classifier.score(X_test, y_test)

        print "SVM-Classifier trained - with score: "
        print score
        print "------------------------------------"
        return clf

    
    """Method to calculate the difference between to coordinates in meter."""
    def calculateError(self, tuple1, tuple2):
        
        radius = 6371000.785
        d_latitude = abs(tuple1[0] - tuple2[0])
        d_longitude = abs(tuple1[1] - tuple2[1])
        return round(radius*math.sqrt((math.pi*d_latitude/180)**2 + (math.pi*d_longitude/180)**2), 2)