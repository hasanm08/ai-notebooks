import numpy as np
import random
import time
class LVQ:
    def __init__(self , dimension , size , class_number):
        self.dimension = dimension
        self.size = size
        a = np.zeros(shape=(class_number , dimension))
        for w in range(class_number):
            vals = []
            for d in range(dimension):
                vals.append(random.randint(0,size))
            a[w] = vals
        self.weightVector = a
        self.data = None

    def load_data(self , input_data_array):
        print("[INFO] Loading data in to LVQ input neurons...")
        if(len(input_data_array) > 0):
            if(len(input_data_array[0].SOMLocation) == self.dimension):
                self.data = input_data_array
                print("[Success] Data loaded into LVQ network Successfully")
                return
            else:
                print("DataError: the dimension of datapoints doesn\'t match with the LVQ network")
                exit()
        else:
            print("DataErrot: No data found")
            exit()

    def second_smallest(self , array):
        m1, m2 = float('inf'), float('inf')
        for x in numbers:
            if x <= m1:
                m1, m2 = x, m1
            elif x < m2:
                m2 = x
        return m2

    def train(self , epoch_num=2 , learning_rate=0.1 , version=1):
        for e in range(epoch_num):
            start_time = time.time()
            print("[Training] LVQ epoch #" , e+1)
            for point in self.data:
                mins = []
                for weight_vector in self.weightVector:
                    mins.append(point.get_distance_from_lvq_weight(weight_vector))
                minIndex = mins.index(min(mins))
                #print(self.weightVector[minIndex] + (learning_rate*(list(point.SOMLocation) - self.weightVector[minIndex])))
                if(version == 1):
                    if(point.label == minIndex+1):
                        self.weightVector[minIndex] = self.weightVector[minIndex] + (learning_rate*(list(point.SOMLocation) - self.weightVector[minIndex]))
                    else:
                        self.weightVector[minIndex] = self.weightVector[minIndex] - (learning_rate*(list(point.SOMLocation) - self.weightVector[minIndex]))
                if(version == 2):
                    if(point.label == minIndex+1):
                        self.weightVector[minIndex] = self.weightVector[minIndex] + (learning_rate*(list(point.SOMLocation) - self.weightVector[minIndex]))
                    else:
                        self.weightVector[minIndex] = self.weightVector[minIndex] - (learning_rate*(list(point.SOMLocation) - self.weightVector[minIndex]))
                        sorted_mins = sorted(mins)
                        for val in sorted_mins[1:]:
                            if(mins.index(val)+1 == point.label):
                                secondMinIndex = mins.index(val)
                                self.weightVector[secondMinIndex] = self.weightVector[secondMinIndex] + (learning_rate*(list(point.SOMLocation) - self.weightVector[secondMinIndex]))
            print("[Training] Epoch #" , e+1 , " time: " ,(time.time() - start_time) , " seconds")
            learning_rate *= 0.5


    def predict(self , test_data_array):
        corrects = 0
        for point in test_data_array:
            mins = []
            for weight_vector in self.weightVector:
                mins.append(point.get_distance_from_lvq_weight(weight_vector))
            minIndex = mins.index(min(mins))
            print(point.label, '---' , minIndex+1)

