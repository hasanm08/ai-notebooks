# %% [markdown]
# #Part A

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
df = pd.read_csv('bank.csv')
df.head()

# %%%%

def labler(value, labels=['very large', 'large', 'fine', 'low', 'very low'], range=[0, 100]):
    res = []
    for item in value:
        percent = range[1]/len(labels)
        value_state = int((item//percent)*-1)
        res.append(labels[value_state])
    return res

# %%%%

def splitter(df,columns,alphas=[2,3,4,5]):
    result = df.copy()
    # without revol.util 85%
    # with revol.util 91%
    for feature_name in columns:
        # print (f"@{feature_name}, Initial Entropy = {entropy(df[feature_name])}")
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        normal_value=((df[feature_name] - min_value) / (max_value - min_value))*100
        res=[]
        for alpha in alphas:
          for item in normal_value:
            percent=100/alpha
            value_state=int((item//percent))
            res.append(value_state)
            pass
          classes=np.unique(res)
          class_items=[]
          for index in classes:
            tmp_items=[]
            for item in res:
              if item ==index:
                tmp_items.append(item)
            class_items.append(tmp_items)
          # print(f"@{feature_name}, alpha ={alpha} : Gain is {gain(df[feature_name],class_items)}")
          # print(f"@{feature_name}, alpha ={alpha} : Entropy is {entropy([len(item) for item in class_items])}")

        #   for item in class_items:
        #     print(f"@{feature_name}, alpha ={alpha} : Entropy is {entropy(item)}")
        # result[feature_name] = labler(value=normal_value)

splitter(df, ['int.rate','installment','log.annual.inc','dti','fico','days.with.cr.line','revol.bal','revol.util'])

#%%%%

def normalize(df):
    # brings all antecedants in [0,100] interval
    result = df.copy()
    # without revol.util 85%
    # with revol.util 91%
    for feature_name in ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        normal_value = ((df[feature_name] - min_value) /
                        (max_value - min_value))*100
        # split all to 5 parts accuracy 91
        # result[feature_name] = labler(value=normal_value)

        #split with min entropy alpha cut acc = 86%
        # result[feature_name] = labler(value=normal_value,labels=['low','high'])

        #split with max gain alpha cut acc =87
        if feature_name == 'revol.bal':
          result[feature_name] = labler(value=normal_value)
        else:
          result[feature_name] = labler(value=normal_value,labels=['low','high'])
    return result

# %%%%

class ID3:
    def fit(self, input, output):
        data = input.copy()
        data[output.name] = output
        self.tree = self.decision_tree(data, data, input.columns, output.name)


    def entropy(self, feature_column):
        #return the Entropy of a probability distribution:
        #entropy(p) = − SUM (Pi * log(Pi) )
        values, counts = np.unique(feature_column, return_counts=True)

        en_list = []
        for i in range(len(values)):
            probability = counts[i]/np.sum(counts)
            en_list.append(-probability*np.log2(probability))
        entropy = np.sum(en_list)

        return entropy

    def information_gain(self, data, feature_name, target_name):
        #return the information gain:
        #gain() = total_entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )

        total_entropy = self.entropy(data[target_name])

        values, counts = np.unique(data[feature_name], return_counts=True)

        # calculate weighted entropy of subset
        weighted_entropy_list = []

        for i in range(len(values)):
            subset_probability = counts[i]/np.sum(counts)
            subset_entropy = self.entropy(data.where(data[feature_name] == values[i]).dropna()[target_name])
            weighted_entropy_list.append(subset_probability*subset_entropy)

        total_weighted_entropy = np.sum(weighted_entropy_list)

        # calculate information gain
        information_gain = total_entropy - total_weighted_entropy

        return information_gain

    def decision_tree(self, data, basic_data, features_name, target_name, parent_node_class=None):
        # base cases:
        unique_classes = np.unique(data[target_name])
        # if data is pure, return the majority class of subset
        if len(unique_classes) <= 1:
            return unique_classes[0]
        # if subset is empty, ie. no samples, return majority class of basic data
        elif len(data) == 0:
            majority_class_index = np.argmax(
                np.unique(basic_data[target_name], return_counts=True)[1])
            return np.unique(basic_data[target_name])[majority_class_index]
        # if data set contains no features to train with, return parent node class
        elif len(features_name) == 0:
            return parent_node_class
        # if none of the above are true, construct a branch:
        else:
            # determine parent node class of current branch
            majority_class_index = np.argmax(
                np.unique(data[target_name], return_counts=True)[1])
            parent_node_class = unique_classes[majority_class_index]

            # determine information gain values for each feature
            # choose feature which best splits the data, ie. highest value
            ig_values = [self.information_gain(data, feature, target_name) for feature in features_name]
            best_feature_index = np.argmax(ig_values)
            best_feature = features_name[best_feature_index]

            # create tree structure, empty at first
            tree = {best_feature: {}}

            # remove best feature from available features, it will become the parent node
            features_name = [i for i in features_name if i != best_feature]
            # create nodes under parent node
            parent_values = np.unique(data[best_feature])

            for value in parent_values:
                sub_data = data.where(data[best_feature] == value).dropna()
                # call the algorithm recursively
                subtree = self.decision_tree(
                    sub_data, basic_data, features_name, target_name, parent_node_class)
                # add subtree to original tree
                tree[best_feature][value] = subtree
            return tree

    def make_prediction(self, sample, tree, default=1):
        # map sample data to tree
        for attribute in list(sample.keys()):
            # check if attribute exists in tree
            if attribute in list(tree.keys()):
                try:
                    result = tree[attribute][sample[attribute]]
                except:
                    return default

                result = tree[attribute][sample[attribute]]

                # if more attributes exist within result, recursively find best result
                if isinstance(result, dict):
                    return self.make_prediction(sample, result)
                else:
                    return result

    def predict(self, input):
        # convert input data into a dictionary of samples
        samples = input.to_dict(orient='records')
        predictions = []

        # make a prediction for every sample
        for sample in samples:
            predictions.append(self.make_prediction(sample, self.tree, 1.0))

        return predictions

# %%%%%
#training and testing

normal_df = normalize(df)
# display(normal_df)

X = normal_df.drop(columns="credit.policy")
y = normal_df["credit.policy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = ID3()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(" ")
print("accuracy_PartA",accuracy_score(y_test, y_pred))
print(" ")



# #Part B
# %%

train_df = pd.read_csv('KDDTrain.csv',  comment='@', header=None)
test_df = pd.read_csv('KDDTest.csv', comment='@', header=None)
protocol_label = preprocessing.LabelEncoder()
service_label = preprocessing.LabelEncoder()
flag_label = preprocessing.LabelEncoder()
class_label = preprocessing.LabelEncoder()
train_df.dropna()
test_df.dropna()
# print(train_df)
# print(test_df)
train_df[1] = protocol_label.fit_transform(train_df[1])
train_df[2] = service_label.fit_transform(train_df[2])
train_df[3] = flag_label.fit_transform(train_df[3])
train_df[41] = class_label.fit_transform(train_df[41])

test_df[1] = protocol_label.transform(test_df[1])
test_df[2] = service_label.transform(test_df[2])
test_df[3] = flag_label.transform(test_df[3])
test_df[41] = class_label.transform(test_df[41])
# print(train_df)
X_train = train_df.drop(columns=41)
X_test = test_df.drop(columns=41)
y_train = train_df[41]
y_test = test_df[41]

# %%
RandomForestClassifier
print('------------RandomForestClassifier---------- ')
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_predict))
print("percision:", precision_score(y_test, y_predict))
print("f1:", f1_score(y_test, y_predict))
print("recall:", recall_score(y_test, y_predict))


# %%
# AdaBoostClassifier
print('------------AdaBoostClassifier---------- ')
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_predict))
print("percision:", precision_score(y_test, y_predict))
print("f1:", f1_score(y_test, y_predict))
print("recall:", recall_score(y_test, y_predict))
