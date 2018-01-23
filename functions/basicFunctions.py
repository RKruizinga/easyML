from collections import Counter
import random

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

### Function to return average of a list
### input(list)
def avg(l):
  return sum(l) / len(l)

### Function to return count per key
### input(list)
def keyCounter(l):
  return Counter(l)

### Function to print evaluation text of a script
### input(Y_test_list, Y_predicted_list, labels_list)
def getMetrics(Y_test, Y_predicted, labels):
  accuracy_count = 0
  for i in range(0, len(Y_predicted)):
    if Y_predicted[i] == Y_test[i]:
      accuracy_count += 1
  accuracy = accuracy_count/len(Y_predicted)

  already_set = False
  clean_labels = [] #to report without errors
  if len(labels) == 1:
    if labels[0] not in Y_predicted:
      precision = 0.0
      recall = 0.0
      f1score = 0.0
      already_set = True
    clean_labels.append(labels[0])
  else:
    for label in labels:
      if label in Y_predicted:
        clean_labels.append(label)

  if already_set == False:
    precision = sklearn.metrics.precision_score(Y_test, Y_predicted, average="macro", labels=clean_labels)
    recall = sklearn.metrics.recall_score(Y_test, Y_predicted, average="macro", labels=clean_labels)
    f1score = sklearn.metrics.f1_score(Y_test, Y_predicted, average="macro", labels=clean_labels)

  return accuracy, precision, recall, f1score

def classifier(method, data):
  if method == 'bayes':
    return Bayes(data) 
  elif method == 'svm':
    classifier = SVM(data) 
  elif method == 'knear':
    classifier = KNeighbors(data)
  elif method == 'tree':
    classifier = DecisionTree(data)
  elif method == 'neural':
    from neuralNetworkClassifier import NeuralNetwork
    classifier = NeuralNetwork(data)
  elif method == 'baseline':
    classifier = Baseline(data)