import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#Class with features specifically for text
class TextFeatures:

### Function to count all words in X
### Input function(X_list)
  class wordCount(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = []
      for x in X:
        newX.append(len(x.split(' ')))
      return np.transpose(np.matrix(newX))

### Function to count all characters in X
### Input function(X_list)
  class characterCount(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X):
      newX = [len(x) for x in X]
      return np.transpose(np.matrix(newX))

### Function to count a specific word in X
### Input function(X_list, string)
  class wordSpecificCount(TransformerMixin):
    def fit(self, X, y=None):
      return self

    def transform(self, X, find = 'word'):
      newX = []
      for x in X:
        user_counter = 0
        tokens = x.split(' ')
        for token in tokens:
          if find in token:
            user_counter += 1
        newX.append(user_counter)
      return np.transpose(np.matrix(newX))