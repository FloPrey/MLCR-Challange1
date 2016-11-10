from DataAnalysis.repository import Repository
from DataAnalysis.configuration import config
from pandas import DataFrame
import numpy as np

class datasetCreator():
    
    repository = Repository(config)
    dataset, labels = repository.get_dataset_and_labels()  
    
    # Replace remaining NaN values with -85
    #dataset = dataset.fillna(-85)
    
    # external file called locations containing all 190 locations 
    with open("locations.txt") as inputFile:
        locationList = [tuple(line.split(',')) for line in inputFile.readlines()]
    
    # create lists containing the floor and coordinate values
    floors = [floor[2].strip() for floor in locationList]
    coordinates = [(float(floor[0]), float(floor[1])) for floor in locationList]

    # Convert coordinate values to numerical labels
    numerical_labels = np.array([repository.locations.keys().index(tuple(l)) for
                        l in labels])
    
    # Y-Value for the learning algorithm --> Floor Labels
    floorList = []
     
    # Fill floor labels in correct order according to the coordinate labels
    for nlabel in numerical_labels:
        floorIndex = coordinates.index(repository.locations.keys()[nlabel])
        floorList.append(floors[floorIndex])
         
    # Floor Labels for the learning algorithm     
    floorLabels = np.array(floorList)
    
    # Change indey from id to the corresponding numerical_label
    dataset = dataset.set_index(numerical_labels)
    
    dataset = dataset.sort_index(axis=0, level=None, ascending=True)
    
    print dataset
    
    # save dataset to csv for further visualization
    #dataset.to_csv('dataset.csv',index=True,header=True)
    
    # count access point's non NaN values and sort output
    #dataset = dataset.count().reset_index(name='count').sort_values(['count'], ascending=True)