import os
import xml.etree.ElementTree as ET

import copy

import re
import pickle 

import random

class Data:

  X_train = []
  Y_train = []
  X_dev = []
  Y_dev = []
  X_test = []

  def __init__(self, language, avoid_skewness = False):
    if language == ['english']:
      self.files = {'train': 'eng-train.pickle', 
                    'development': 'eng-trial.pickle',
                    'test': 'us_test.pickle' }
    elif language == ['spanish']:
      self.files = {'train': 'es-train.pickle', 
                    'development': 'es-trial.pickle',
                    'test': 'es_test.pickle' }

    self.avoid_skewness = avoid_skewness
    self.readFiles()

  def readFile(self, file):
    return pickle.load(open(self.files[file], 'rb'))

  def transform(self):
    for Y, X in self.data_train:

      X = re.sub(r'\…', ' $INSTAGRAM$', X)
      X = re.sub(r'http:\/\/t.co\S*', ' %URL% ', X)
      X = re.sub(r'https:\/\/t.co\S*', ' %URL% ', X)

      X = re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)
      X = re.sub(r'(.)\1{3,}', r'\1\1\1', X)
      
      self.X_train.append(X)
      self.Y_train.append(Y)

    for Y, X in self.data_dev:

      X = re.sub(r'\…', ' $INSTAGRAM$', X)
      X = re.sub(r'http:\/\/t.co\S*', ' %URL% ', X)
      X = re.sub(r'https:\/\/t.co\S*', ' %URL% ', X)
      
      X = re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)
      X = re.sub(r'(.)\1{3,}', r'\1\1\1', X)

      self.X_dev.append(X)
      self.Y_dev.append(Y)

    for X in self.data_test:

      X = re.sub(r'\…', ' $INSTAGRAM$', X)
      X = re.sub(r'http:\/\/t.co\S*', ' %URL% ', X)
      X = re.sub(r'https:\/\/t.co\S*', ' %URL% ', X)
      
      X = re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)
      X = re.sub(r'(.)\1{3,}', r'\1\1\1', X)

      self.X_test.append(X)

    if self.avoid_skewness:
      self.getSubsetUnskewedTrain()

    self.X = copy.copy(self.X_train)
    self.Y = copy.copy(self.Y_train)

    self.X.extend(self.X_test)
    self.Y.extend(self.Y_test)

    self.labels = list(set(self.Y))
    self.split_amount = len(self.X_train)

  def readFiles(self):
    self.data_train = self.readFile('train')
    self.data_dev = self.readFile('development')
    self.data_test = self.readFile('test')

    self.transform()

  def subset(self, train_amount, test_amount):
    self.X_train, self.Y_train = self.subsetBase(self.X_train, self.Y_train, train_amount)
    self.X_test, self.Y_test = self.subsetBase(self.X_test, self.Y_test, test_amount)

  def subsetBase(self, list_a, list_b, amount):
    all_keys = []
    for i, val in enumerate(list_a):
      all_keys.append(i)

    new_list_a = []
    new_list_b = []

    subset_keys = sorted(random.sample(all_keys, amount))

    for key in subset_keys:
      new_list_a.append(list_a[key])
      new_list_b.append(list_b[key])

    return new_list_a, new_list_b

  ### Function to make train dataset unskewed
  ### input(X_train_list, Y_train_list, Y_train_raw_list)
  def getSubsetUnskewedTrain(self, X_train, Y_train):
    data_distribution = BasicFunctions.keyCounter(Y_train)
    Y = Y_train

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

    self.X_train = new_X_train
    self.Y_train = new_Y_train



  

