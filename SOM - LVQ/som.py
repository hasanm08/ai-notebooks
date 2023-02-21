import random
import numpy as np
import time
np.set_printoptions(precision=20)
class SOM:
    def __init__(self , dimension , size , neuron_dimension):
        self.dimension = dimension
        self.size = size
        self.neuron_dimension =  neuron_dimension
        t = ()
        for d in range(dimension):
             t = t + (size,)
        t = t + (neuron_dimension,)
        a=np.random.rand(neuron_dimension*(size**dimension))
        self.net = a.reshape(t)
        self.net = self.net.astype(np.float , copy = True)
        self.data = None

    def load_input_data(self , input_data_array):
        print("[INFO] Loading data in to SOM input neurons...")
        if(len(input_data_array) > 0):
            if(input_data_array[0].dimension == self.neuron_dimension):
                self.data = input_data_array
                print("[Success] Data loaded into SOM network Successfully")
                return
            else:
                print("DataError: the dimension of datapoints doesn\'t match with the SOM network")
                exit()
        else:
            print("DataErrot: No data found")
            exit()

    def search(self , x):
        try:
            #s.net[t[0:len(t)-1]]
            return list(zip(*np.where(self.net==x)))
        except IndexError:
            pass

    def fit(self , epoch_num=2 , learning_rate=0.1 , neighbor_function = None):
        for epoch in range(epoch_num):
            start_time = time.time()
            print("[Training] SOM epoch #" , epoch+1)
            weights = self.net.reshape((self.size**self.dimension , self.neuron_dimension))
            for point in self.data :
                mins = []
                for weightV in weights:
                    mins.append(point.get_distance_from_node(weightV))
                minIndex = mins.index(min(mins))
                search_result = self.search(weights[minIndex])
                point.SOMLocation = search_result[0][0:self.dimension]
                #Updating the winner nueron
                delta = (learning_rate*(point.featureVector - weights[minIndex]))
                self.update_node(delta , search_result[0][0:self.dimension])
                #updating neighbors
                if(neighbor_function == "square"):
                    try:
                        winner_location = list(search_result[0][0:self.dimension])
                        for i in range(len(winner_location)):
                            temp = winner_location
                            temp[i] = temp[i]+1
                            neighbor_location = tuple(temp)
                            self.update_node(delta , neighbor_location)
                            for j in range(i+1,len(winner_location)-1):
                                temp[j] = temp[j] + 1
                                neighbor_location2 =  tuple(temp)
                                self.update_node(delta , neighbor_location2)
                        for i in range(len(winner_location)):
                            temp = winner_location
                            temp[i] = temp[i]-1
                            neighbor_location = tuple(temp)
                            self.update_node(delta , neighbor_location)
                            for j in range(i+1,len(winner_location)-1):
                                temp[j] = temp[j] -1
                                neighbor_location2 =  tuple(temp)
                                self.update_node(delta , neighbor_location2)
                    except:
                        pass
                
            print("[Training] Epoch #" , epoch+1 , " time: " ,(time.time() - start_time) , " seconds")
        return
        #print(weights1 - weights)
    def update_node(self , delta , location):
        delta = delta.astype(np.float , copy = True)
        for i in range(len(delta)):
            #print(self.net[location][i] , "old" ,location)
            self.net[location][i] += delta[i]
            #print(self.net[location][i] , "new" , location)
        return
            

    def predict(self , test_data_array):
        weights = self.net.reshape((self.size**self.dimension , self.neuron_dimension))
        for point in test_data_array :
            mins = []
            for weightV in weights:
                mins.append(point.get_distance_from_node(weightV))
                minIndex = mins.index(min(mins))
                #print(weights[minIndex])
                search_result = self.search(weights[minIndex])
                point.SOMLocation = search_result[0][0:self.dimension]
        return

    
    def iterate(self):
        a = self.net.reshape((self.size**self.dimension , self.neuron_dimension))
        return a
        
