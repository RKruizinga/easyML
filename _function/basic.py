from collections import Counter
import random
import numpy as np

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import confusion_matrix

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import pprint

pp = pprint.PrettyPrinter(indent=2)

#Function to run our system, just for a clean main.py, as we do not want to change this.
def run(k, method, data, features, printer, predict_method, new_classifier=None, print_details=1, show_fitting=False):
  if k > 1:
    from _class.validation import KFoldValidation
    kfold = KFoldValidation(k, method, data, features, predict_method, new_classifier, print_details, show_fitting)

    if print_details >= 1:
      kfold.printBasicEvaluation()
    return kfold

  else:
    c = classifier(method, data, predict_method, show_fitting)
    c.classify(features, new_classifier)
    c.evaluate()
    if print_details >= 1:
      c.printBasicEvaluation()
    if print_details >= 2:
      c.printClassEvaluation()

    #writeResults(options.args.predict_languages, classifier.Y_development, classifier.Y_development_predicted, 'development')
    if print_details >= 3:
      printer.confusionMatrix(c.Y_development, c.Y_development_predicted, data.labels)
    return c
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
def classificationMetrics(Y_test, Y_predicted, labels):
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

### Function to print evaluation text of a script
### input(Y_test_list, Y_predicted_list, labels_list)
def regressionMetrics(Y_test, Y_predicted, labels=None):

  # for key, Y_test_i in enumerate(Y_test):
  #   print(Y_test_i, Y_predicted[key])
  r2score = sklearn.metrics.r2_score(Y_test, Y_predicted)
  mean_abs_err = sklearn.metrics.mean_absolute_error(Y_test, Y_predicted)
  mean_squ_err = sklearn.metrics.mean_squared_error(Y_test, Y_predicted)
  mutual_info = sklearn.metrics.mutual_info_score(Y_test, Y_predicted)

  return mean_abs_err, mean_squ_err, r2score, mutual_info

def classifier(method, data, predict_method, show_fitting):
  if method == 'bayes':
    from classifier.naiveBayes import NaiveBayes
    return NaiveBayes(data, predict_method, show_fitting) 
  elif method == 'svm':
    from classifier.svm import SVM
    return SVM(data, predict_method, show_fitting) 
  elif method == 'knear':
    from classifier.kNeighbors import KNeighbors
    return KNeighbors(data, predict_method, show_fitting)
  elif method == 'tree':
    from classifier.decisionTree import DecisionTree
    return DecisionTree(data, predict_method, show_fitting)
  elif method == 'neural' or method == 'nn':
    from classifier.neuralNetwork import NeuralNetwork
    return NeuralNetwork(data, predict_method, show_fitting)
  elif method == 'baseline':
    from classifier.baseline import Baseline
    return Baseline(data, predict_method, show_fitting)
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


def printProbabilities(Y_test, Y_predicted_proba):
  no = []
  yes = []
  no_x = []
  yes_x = []
  for key, Y_test_i in enumerate(Y_test): 
    if Y_test_i is 1:
      yes.append(round(Y_predicted_proba[key][1], 3))
      yes_x.append(random.randint(0,100))

    else:
      no.append(round(Y_predicted_proba[key][1], 3))
      no_x.append(random.randint(0,100))
      
  N = 500
  plotly.tools.set_credentials_file(username='RKruizinga', api_key='twDl3OBUY9wXbwTYUdkJ')

  trace0 = go.Scatter(
    x = yes_x,
    y = yes,
    name = 'Converted',
    mode = 'markers',
    marker = dict(
      size = 10,
      color = 'rgba(0, 255, 0, .9)',
      line = dict(
        width = 2,
        color = 'rgb(0, 0, 0)'
      )
    )
  )

  trace1 = go.Scatter(
    x = no_x,
    y = no,
    name = 'Not Converted',
    mode = 'markers',
    marker = dict(
      size = 10,
      color = 'rgba(255, 0, 0, .2)',
      line = dict(
        width = 2,
      )
    )
  )

  data = [trace0, trace1]

  layout = dict(title = 'Avg Conversion Matrix',
        yaxis = dict(zeroline = False),
        xaxis = dict(zeroline = False)
        )

  fig = dict(data=data, layout=layout)
  py.plot(fig, file_id='https://plot.ly', filename='avg-conversion-rate-regression')