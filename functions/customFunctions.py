from collections import Counter
import random
import os
import time
import numpy as np

### Custom feature to find the language of shortened text
### input(Y_test_list, Y_predicted_list, labels_list)
def getLanguages(argument_languages):
  possible_languages = ['dutch', 'english', 'spanish', 'italian']
  if argument_languages == 'all':
    predict_languages = possible_languages

  predict_languages = argument_languages.split(',')

  new_format = []
  if len(predict_languages) == 1:
    if predict_languages[0] not in possible_languages:
      for letter in predict_languages[0]:
        for possible_language in possible_languages:
          if letter == possible_language[0]:
            new_format.append(possible_language)
      predict_languages = new_format
  return predict_languages

### Function to write results to file
### input()
def writeResults(languages, Y_test, Y_predicted):
  language = languages[0]
  cur_date = time.strftime("%Y_%U")
  cur_time = time.strftime("%H_%M_%S")

  if not os.path.exists(str(cur_date)):
    os.makedirs(str(cur_date))
  
  if not os.path.exists(str(str(cur_date) + '/' + str(cur_time))):
    os.makedirs(str(str(cur_date) + '/' + str(cur_time)))

  output_test = open(str(cur_date) + '/' + str(cur_time) + '/' + language + '_goldDEV.output.txt', 'w')
  for i in Y_test:
    output_test.write(i)
    output_test.write("\n")

  output_predicted = open(str(cur_date) + '/' + str(cur_time) + '/' + language + '_predicted.output.txt', 'w')
  for i in Y_predicted:
    output_predicted.write(i)
    output_predicted.write("\n") 

### Function to print confusion matrix of classes
### input(Y_test_list, Y_predicted_list, labels_list)
def writeConfusionMatrix(Y_test, Y_predicted):
  from createConfusionMatrix import main as ConfusionMatrix
  ConfusionMatrix(Y_test, Y_predicted)