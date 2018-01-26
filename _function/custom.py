from collections import Counter
import random
import os
import time
import numpy as np
import pickle

### Custom feature to find the language of shortened text
### input(Y_test_list, Y_predicted_list, labels_list)
def languages(argument_languages):
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
def writeResults(language, Y, Y_predicted, name):
  cur_date = time.strftime("%Y_%m")
  cur_time = time.strftime("%H_%M_%S")

  if not os.path.exists('output/'+str(cur_date)):
    os.makedirs('output/'+str(cur_date))
  
  if not os.path.exists('output/'+str(str(cur_date) + '/' + str(cur_time))):
    os.makedirs('output/'+str(str(cur_date) + '/' + str(cur_time)))

  Y_output = open('output/'+str(cur_date) + '/' + str(cur_time) + '/' + language + '_gold_'+name+'.output.txt', 'w')
  for i in Y:
    Y_output.write(i)
    Y_output.write("\n")

  Y_predicted_output = open('output/'+str(cur_date) + '/' + str(cur_time) + '/' + language + '_predicted'+name+'.output.txt', 'w')
  for i in Y_predicted:
    Y_predicted_output.write(i)
    Y_predicted_output.write("\n") 

def writeWordEmbeddings(word_embeddings, word_embeddings_index, var1, var2):
  cur_date = time.strftime("%Y_%m")
  cur_time = time.strftime("%H")

  if type(var1) == list:
    var1 = '_'.join(var1)

  if not os.path.exists('output/word_embeddings'):
    os.makedirs('output/word_embeddings')

  if not os.path.exists('output/word_embeddings/'+str(cur_date)):
    os.makedirs('output/word_embeddings/'+str(cur_date))
  
  if not os.path.exists('output/word_embeddings/'+str(str(cur_date) + '/' + str(cur_time))):
    os.makedirs('output/word_embeddings/'+str(str(cur_date) + '/' + str(cur_time)))

  open_file_we = open('output/word_embeddings/'+str(cur_date) + '/' + str(cur_time) + '/' + var1 +'_'+ var2 + '_word_embeddings.pickle', 'wb') 
  pickle.dump(word_embeddings, open_file_we)
  open_file_wei = open('output/word_embeddings/'+str(cur_date) + '/' + str(cur_time) + '/' + var1 +'_'+ var2 + '_word_embeddings_index.pickle', 'wb') 
  pickle.dump(word_embeddings_index, open_file_wei)

def readWordEmbeddings(var1, var2):
  cur_date = time.strftime("%Y_%m")
  cur_time = time.strftime("%H")

  if type(var1) == list:
    var1 = '_'.join(var1)

  try:
    open_file_we = open('output/word_embeddings/'+str(cur_date) + '/' + str(cur_time) + '/' + var1 +'_'+ var2 +'_word_embeddings.pickle', 'rb')
    open_file_wei = open('output/word_embeddings/'+str(cur_date) + '/' + str(cur_time) + '/' + var1 +'_'+ var2 +'_word_embeddings_index.pickle', 'rb')
    
    return pickle.load(open_file_we), pickle.load(open_file_wei)
  except:
    return None, None

### Function to print confusion matrix of classes
### input(Y_test_list, Y_predicted_list, labels_list)
def writeConfusionMatrix(Y_test, Y_predicted):
  from createConfusionMatrix import main as ConfusionMatrix
  ConfusionMatrix(Y_test, Y_predicted)