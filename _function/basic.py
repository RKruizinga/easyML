from collections import Counter
import random
import numpy as np

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#Function to run our system, just for a clean main.py, as we do not want to change this.
def run(k, method, data, features, printer, new_classifier=None, print_details=1, show_fitting=False):
  if k > 1:
    from _class.validation import KFoldValidation
    kfold = KFoldValidation(k, method, data, features, new_classifier, print_details, show_fitting)

    if print_details >= 1:
      kfold.printBasicEvaluation()

  else:
    c = classifier(method, data, show_fitting)
    c.classify(features, new_classifier)
    c.evaluate()
    if print_details >= 1:
      c.printBasicEvaluation()
    if print_details >= 2:
      c.printClassEvaluation()

    #writeResults(options.args.predict_languages, classifier.Y_development, classifier.Y_development_predicted, 'development')
    if print_details >= 3:
      printer.confusionMatrix(c.Y_development, c.Y_development_predicted, data.labels)

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
def metrics(Y_test, Y_predicted, labels):
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

def classifier(method, data, show_fitting):
  if method == 'bayes':
    from classifier.naiveBayes import NaiveBayes
    return NaiveBayes(data, show_fitting) 
  elif method == 'svm':
    from classifier.svm import SVM
    return SVM(data, show_fitting) 
  elif method == 'knear':
    from classifier.kNeighbors import KNeighbors
    return KNeighbors(data, show_fitting)
  elif method == 'tree':
    from classifier.decisionTree import DecisionTree
    return DecisionTree(data, show_fitting)
  elif method == 'neural':
    from classifier.neuralNetwork import NeuralNetwork
    return NeuralNetwork(data, show_fitting)
  elif method == 'baseline':
    from classifier.baseline import Baseline
    return Baseline(data, show_fitting)
  else:
    return 'Not a valid classification method!'

def unskewedTrain(X_train, Y_train, Y_train_raw = None):
  if Y_train_raw == None:
    data_distribution = keyCounter(Y_train)
    Y = Y_train
  else:
    data_distribution = keyCounter(Y_train_raw)
    Y = Y_train_raw

  lowest_amount = 0
  for label in data_distribution:
    if data_distribution[label] < lowest_amount or lowest_amount == 0:
      lowest_amount = data_distribution[label]
  key_dict = {}
  for i, label in enumerate(Y):
    if label not in key_dict:
      key_dict[label] = [i]
    else:
      key_dict[label].append(i)

  new_X_train = []
  new_Y_train = []
  all_keys = []
  new_dict = {}
  for label in key_dict: 
    new_dict[label] = random.sample(key_dict[label], lowest_amount)
    all_keys += new_dict[label]
  for i in sorted(all_keys):
    new_X_train.append(X_train[i])
    new_Y_train.append(Y_train[i])

  return new_X_train, new_Y_train
