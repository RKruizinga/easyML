from collections import Counter
import random
import os
import time
import datetime
import numpy as np
import re
import collections

import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from functions.basicFunctions import keyCounter

class Printer:

  def __init__(self, scope):
    self.start_time = datetime.datetime.now()
    self.scope = scope
    self.start()


  def tabFill(self, text, value = None):
    if value == None:
      if len(text) > 20:
        return '\t\t\t\t\t\t' #6 tabs if 21 characters
      else:
        return '\t\t\t'
    else: 
      if len(str(value)) > 20:
        return '\t'
      if len(str(value)) > 15:
        return '\t\t'
      elif len(str(value)) > 5:
        return '\t\t\t'
      else:
        return '\t\t\t\t'

  def tabSpace(self, text):
    if len(text) > 13:
      return '\t\t'
    elif len(text) > 5:
      return '\t\t\t'
    else:
      return '\t\t\t\t'

### Function to print standard text of a script
### input(dictionary(args_name, args_value))
  def system(self, args):

    text = args['title'] + ' Output'

    print('#'*91)
    print('{} \t {} {} {}'.format('#'*10, text, self.tabFill(text), '#'*10))

    for name in args:
      if name != 'title':
        value = args[name]
        if type(value) == list:
          value = ', '.join(value).title()
        name = re.sub(r'_', ' ', name)

        tags = '#'*10
        middle_tabs = '\t'
        end_tabs = '\t\t'

        print('{} \t {} {} {} {} {}'.format(tags, name.title(), self.tabSpace(name), value, self.tabFill(name, value), tags))
    print('#'*91)

### Function to print evaluation text of a script
### input(accuracy_float, precision_float, recall_float, f1score_float, text_string)
  def evaluation(self, accuracy, precision, recall, f1score, text):    
    print("~~~" + text + "~~~ \n")
    print("Accuracy:\t {}".format(round(accuracy, 3)))
    print("Precision:\t {}".format(round(precision, 3)))
    print("Recall:\t\t {}".format(round(recall, 3)))
    print("F1-Score:\t {}".format(round(f1score, 3)))


### Function to print evaluation text of a script
### input(Y_test_list, Y_predicted_list, labels_list)
  def classEvaluation(self, Y_test, Y_predicted, labels):

    print("\n~~~ Class Evaluation ~~~ \n")
    print("Class \t Precision \t Recall \t F-score")

    for label in labels:
      accuracy, precision, recall, f1score = getMetrics(Y_test, Y_predicted, [label])
      print('{} \t {} \t\t {} \t\t {}'.format(
        label,
        round(precision, 3),
        round(recall, 3),
        round(f1score, 3)
      ))

### Function to print label distribution of data
### input(Y_list)
  def labelDistribution(self, Y, text, orderBy='label'):
    label_distribution = keyCounter(Y)

    is_digit = False
    label_distribution_ordered = []
    for label in label_distribution:
      if label.isdigit():
        label_distribution_ordered.append((int(label), label_distribution[label]))
      else:
        label_distribution_ordered.append((label, label_distribution[label]))

    if orderBy == 'label':
      num = 0
      rev = False
    else:
      num = 1
      rev = True
    label_distribution_ordered = sorted(label_distribution_ordered, key=lambda x: x[num], reverse=rev)

    print('~~~Label Distribution of {}~~~'.format(text))
    for label, amount in label_distribution_ordered:
      print('{} \t {}'.format(label, amount))

### Function to print duration of script
### input(start, end)
  def start(self):
    print('{} start time: {}'.format(self.scope, self.start_time.strftime('%Y-%m-%d, %H:%M:%S')))

### Function to print duration of script
### input(start, end)
  def duration(self):
    self.end_time = datetime.datetime.now()
    self.total_time = self.end_time - self.start_time
    time = divmod(total_time.days * 86400 + total_time.seconds, 60)
    self.minutes = time[0]
    self.seconds = time[1]
    print('{}. Run time: {} minutes and {} seconds'.format(self.scope, self.minutes, self.seconds))
    print('{} end time: {}'.format(self.scope, self.end_time.strftime('%Y-%m-%d, %H:%M:%S')))