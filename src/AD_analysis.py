import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import AD_utils

class AD:
    #Initialization of this class takes in three parameters
    # the base parameter is the base data that you want to compare other compounds to
    #base parameter either takes in an array of smiles or a pandas dataframe consisting of only the fingerprints you want to compare
    
    def __init__(self, base, threshold = 0.8, base_type = 'smiles'):
        self.base = base
        self.threshold = threshold
        self.base_type = base_type

        if(self.base_type == 'smiles'):
            #if your self.base is not an array, dont create class
            if(not isinstance(self.base, list)):
                raise TypeError("'base' should be an array for a base_type of 'smiles'")
            
            #if an array, let's compute descriptors and save as ExplicitBitVect
            fps = AD_utils.compute_morganfps(self.base)
            self.base = fps
        
        if(self.base_type == "fingerprint"):
            #check if self.base is a pandas dataframe
            if(not isinstance(self.base, pd.DataFrame)):
                raise TypeError("'base' should be a pandas Dataframe for type fingerprint")


    #function to redefine the various parameters
    def setParams(self, base, threshold, base_type):
        if (base):
            self.base = base
        if(threshold):
            self.threshold = threshold
        if (base_type):
            self.base_type = base_type

    def base_similarity(self, heatmap = True):
        #return the similarites of all compounds in your base
        length = len(self.base)

        array = np.zeros((length, length))
        
        for i in range(len(self.base)):
            for j in range(i, len(self.base)):
                array[i][j] = AD_utils.get_tanimomto_similarities(self.base[i], self.base[j])
                array[j][i] = array[i][j]
        
        if (heatmap):
            sns.heatmap(array, annot=True)
            plt.show()
            return
        
        return array.to_list()


    #compare between test vector and train
    def get_similarity(self, test, return_type = 'max'):
        return_types = ['min' , 'max']

        if (return_type not in return_types):
            raise NameError(f"return_type should be either 'min', 'max', 'all'")
        
        if(isinstance(test, list)):
            return_type = 'max'
        elif(isinstance(test, str)):
            test = [test]
        else:
            raise TypeError('test must be of type "list" or "string"', type(test))
        
        #return the similarity of your test to base
        testbitVect = AD_utils.compute_morganfps(test)

        answer_array = []

        for i in testbitVect:
            similarities = AD_utils.get_bulk_tanimoto_distance(i, self.base)
            answer_array.append(max(similarities))

        return answer_array

    #plotting distance against compound ID
    def plot_distance(self, test, threshold = None, input_info = None):
        if (not isinstance(threshold, (float, int)) and not threshold == None):
            raise TypeError('threshold must be of type integer')
        
        def dist(p):
            return 1-p

        #additional info is a type smile
        #converting it to find its distance.
        input_distance = list(map(dist, self.get_similarity(input_info)))

        #generating distance from calculated similarity
        distance = list(map(dist, self.get_similarity(test)))
        fig, ax = plt.subplots()

        #plot of distance
        sns.scatterplot(distance, ax=ax, color='orange', alpha = 0.5, label='test-data')

        #input info is any additional points that should be plotted on the base distribution
        if input_info:
            sns.scatterplot(x = [300], y=[input_distance[0]], color='blue', ax=ax, label='input')
        
        
        #plot threshold if threshold value is given
        if(threshold):
            plt.axhline(y=threshold, color='r', linestyle='-', linewidth=2, label = 'threshold')

            #boolean to show which distances are in the threshold
            isWithin = list(map(lambda p: p < threshold, input_distance))
        else:
            isWithin = None


        ax.set_title('Applicability domain')
        ax.set_xlabel('Compound ID')
        ax.set_ylabel('Tanimoto distance')
        ax.legend()

        return fig, ax, isWithin