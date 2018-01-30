import os
import xml.etree.ElementTree as ET

import copy

import re
import pickle 

import random

from text.preprocessing import TextPreprocessing
from _function.basic import keyCounter
from sklearn.model_selection import train_test_split

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

  def __init__(self, avoid_skewness = False, data_folder='data/', response_variable='variable', data_method=1):

    self.avoid_skewness = avoid_skewness
    self.data_folder = data_folder
    self.response_variable = response_variable
    self.data_method = data_method

  def preprocessor(self, X, preprocessing):
    if preprocessing != None:
      for method in preprocessing:
        X = TextPreprocessing.run(X, method)

    return X

  def transformByYXrow(self, preprocessing):
    if self.file_train != '' and self.file_development == '':
      X_all = []
      Y_all = []
      for Y, X in self.train:
        X_all.append(X)
        Y_all.append(Y)
        self.preprocessor(X, preprocessing)
  
      self.X_train, self.X_development, self.Y_train, self.Y_development = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)
      self.X_test = self.X_development
      self.Y_test = self. Y_development
      
    else:
      if self.file_train != '':
        for Y, X in self.train:
          self.X_train.append(X)
          self.Y_train.append(Y)
          self.preprocessor(X, preprocessing)

      if self.file_development != '':
        for Y, X in self.development:
          self.X_development.append(X)
          self.Y_development.append(Y)
          if self.file_test == '':
            self.X_test.append(X)
          self.preprocessor(X, preprocessing)
          
      if self.file_test != '':
        for X in self.test:
          self.X_test.append(X)
          self.preprocessor(X, preprocessing)

    if self.avoid_skewness:
        self.X_train, self.Y_train = self.getSubsetUnskewedTrain(self.X_train, self.Y_train)

    self.amount_train = len(self.X_train) 
    self.amount_development = len(self.X_development)  
    self.amount_test = len(self.X_test)  

  def initialize(self, X_train, Y_train, X_development, Y_development):

    self.X_train = X_train
    self.Y_train = Y_train
    self.X = copy.copy(self.X_train)
    self.Y = copy.copy(self.Y_train)
  
    self.X_development = X_development
    self.Y_development = Y_development
    self.X.extend(self.X_development)
    self.Y.extend(self.Y_development)
    
    self.X_test = X_development
    self.X.extend(self.X_test)

    self.amount_train = len(self.X_train) 
    self.amount_development = len(self.X_development)  
    self.amount_test = len(self.X_test)  

  ## transform file(s) to Y, X
  def transform(self, _type='YXrow', preprocessing=None):
    if _type == 'YXrow':
      self.transformByYXrow(preprocessing=preprocessing)
    else:
      pass
    
    self.createXY()
    self.labels = list(set(self.Y)) 

    try:
      if int(self.labels[0]):
        self.labels = sorted(self.labels, key=int)
      else:
        self.labels = sorted(self.labels)
    except:
      self.labels = sorted(self.labels)

  def createXY(self):
    self.X = copy.copy(self.X_train)
    self.Y = copy.copy(self.Y_train)
    self.X.extend(self.X_development)
    self.Y.extend(self.Y_development)
    self.X.extend(self.X_test)

  def load(self, file_name, format):
    if format == 'pickle':
      return pickle.load(open('./'+self.data_folder+file_name, 'rb'))
    elif format == 'specific_age_gender':
      return self.ageGenderLoad(folder=file_name)

  def ageGenderLoad(self, folder):
    documents = {}
    labels = {}

    for language in self.languages:
      folder_path = folder+language

      files = os.listdir(folder_path)

      if language not in documents:
        documents[language] = {}
        
      if language not in labels:
        labels[language] = {}

      for file in files:
        if file.endswith(".xml"):
            tree = ET.parse(folder_path + '/' +file)
            root = tree.getroot()
            for child in root: #get all utterances of a person
              if child.tag == 'document':
                if file[:-4] in documents[language]:
                  documents[language][file[:-4]].append(child.text)
                else:
                  documents[language][file[:-4]] = [child.text]
        if file.endswith(".txt"):
          with open(folder_path + '/' +file) as f:
            lines = f.read()
            lines = lines.split('\n')
            for line in lines:
              if line != '':
                line = line.split(':::')
                labels[language][line[0]] = (line[1], line[2]) # tuple, format: (gender, age)

    X = []
    Y = []
    for language in self.languages:
      for user in documents[language]:
        if self.data_method == 1:
          for document in documents[language][user]:
            X.append(document)
            if self.response_variable == 'gender':
              Y.append(labels[language][user][0])
            else:
              Y.append(labels[language][user][1])
        elif self.data_method == 2:
          X,append('. '.join(documents[language][user]))
          if self.response_variable == 'gender':
            Y.append(labels[language][user][0])
          else:
            Y.append(labels[language][user][1])
    return list(zip(Y, X))

  def subset(self, train_size=50000, development_size=None, test_size=None):
    if train_size != None:
      self.X_train, self.Y_train = self.subsetLists(self.X_train, self.Y_train, train_size)
      self.amount_train = len(self.X_train)
    if development_size != None:
      self.X_development, self.Y_development = self.subsetLists(self.X_development, self.Y_development, development_size)
      self.amount_development = len(self.X_development) 
    if test_size != None:
      self.X_test, self.Y_test = self.subsetLists(self.X_test, self.Y_test, test_size)
      self.amount_test = len(self.X_test) 
    self.createXY()

  def subsetLists(self, list_a, list_b, amount):
    all_keys = []
    for i, val in enumerate(list_a):
      all_keys.append(i)

    new_list_a = []
    new_list_b = []

    subset_keys = sorted(random.sample(all_keys, amount))

    for key in subset_keys:
      new_list_a.append(list_a[key])
      if len(list_b) > 0:
        new_list_b.append(list_b[key])

    return new_list_a, new_list_b

  ### Function to make train dataset unskewed
  ### input(X_train_list, Y_train_list, Y_train_raw_list)
  def getSubsetUnskewedTrain(self, X_train, Y_train):
    data_distribution = keyCounter(Y_train)
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

    return new_X_train, new_Y_train



  

