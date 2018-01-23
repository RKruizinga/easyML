import os
import xml.etree.ElementTree as ET

import copy

import re
import pickle 

import random

from text.textPreprocessing import TextPreprocessing

class Data:

  file_train = ''
  file_development = ''
  file_test = ''

  X_train = []
  Y_train = []
  X_development = []
  Y_development = []
  X_test = []
  Y_test = []

  def __init__(self, avoid_skewness = False, data_folder='data/'):

    self.avoid_skewness = avoid_skewness
    self.data_folder = data_folder

  def preprocessor(self, X, preprocessing):
    if preprocessing != None:
      for method in preprocessing:
        X = TextPreprocessing.run(X, method)

    return X

  def transformByYXrow(self, preprocessing):

    if self.file_train != '':
      for Y, X in self.train:
        self.X_train.append(X)
        self.Y_train.append(Y)
        self.preprocessor(X, preprocessing)
      self.amount_train = len(self.X_train) 

    if self.avoid_skewness:
      self.getSubsetUnskewedTrain()

    if self.file_development != '':
      for Y, X in self.development:
        self.X_development.append(X)
        self.Y_development.append(Y)
        self.preprocessor(X, preprocessing)
      self.amount_development = len(self.X_development)  
        
    if self.file_test != '':
      for X in self.test:
        self.X_test.append(X)
        self.preprocessor(X, preprocessing)
      self.amount_test = len(self.X_test)  

  ## transform file(s) to Y, X
  def transform(self, _type='YXrow', preprocessing=None):
    if _type == 'YXrow':
      self.transformByYXrow(preprocessing=preprocessing)
    else:
      pass
    
    self.createXY()
    self.labels = list(set(self.Y)) 

  def createXY(self):
    if self.file_train != '':
      self.X = copy.copy(self.X_train)
      self.Y = copy.copy(self.Y_train)
  
    if self.file_development != '':
      self.X.extend(self.X_development)
      self.Y.extend(self.Y_development)
    
    if self.file_test != '':
      self.X.extend(self.X_test)

  def load(self, file_name, format):
    if format == 'pickle':
      return pickle.load(open('./'+self.data_folder+file_name, 'rb'))

  def subset(self, train_size=50000, development_size=None, test_size=None):
    if train_size != None:
      self.X_train, self.Y_train = self.subsetLists(self.X_train, self.Y_train, train_amount)
    if development_size != None:
      self.X_development, self.Y_development = self.subsetLists(self.X_development, self.Y_development, development_size)
    if test_size != None:
      self.X_test, self.Y_test = self.subsetLists(self.X_test, self.Y_test, test_amount)

  def subsetLists(self, list_a, list_b, amount):
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



  

