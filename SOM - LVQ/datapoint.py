class DataPoint:
    def __init__(self , feature_vector , label):
        self.dimension = len(feature_vector)
        self.featureVector = feature_vector
        self.label = label 
        self.SOMLocation = None


    def get_distance_from_node(self , weight_vector):
        if(self.dimension == len(weight_vector)):
            dist = 0
            for i in range(self.dimension):
                dist += (self.featureVector[i]-weight_vector[i])**2
            return dist
        else:
            print("DataError: Weights dont match")
            exit()

    def get_distance_from_lvq_weight(self , weight_vector):
        if(len(self.SOMLocation) == len(weight_vector)):
            dist = 0
            for i in range(len(weight_vector)):
                dist += (self.SOMLocation[i] - weight_vector[i])**2
            return dist
        
        else:
            print("DataError: Weights dont match")
            exit()
