import re
#Class with preprocessing specifically for text
class TextPreprocessing:
  def run(X, method):
    if method == 'replaceTwitterInstagram':
      return TextPreprocessing.replaceTwitterInstagram(X)
    elif method == 'replaceTwitterURL':
      return TextPreprocessing.replaceTwitterURL(X)
    elif method == 'replaceSpecialCharacters':
      return TextPreprocessing.replaceSpecialCharacters(X)
    elif method == 'maxCharacterSequence':
      return TextPreprocessing.maxCharacterSequence(X)
    elif method == 'replaceDate':
      return TextPreprocessing.maxCharacterSequence(X)
    elif method == 'replaceYear':
      return TextPreprocessing.maxCharacterSequence(X)
    else:
      return X

  def replaceTwitterInstagram(X):
    return re.sub(r'\…', ' INSTAGRAM', X)

  def replaceTwitterURL(X):
    X = re.sub(r'http:\/\/t.co\S*', 'URL ', X)
    return re.sub(r'https:\/\/t.co\S*', 'URL ', X)

  def replaceSpecialCharacters(X):
    return re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)

  def maxCharacterSequence(X):
    return re.sub(r'(.)\1{2,}', r'\1\1\1', X)
  
  def replaceDate(X):
    return re.sub(r'\d+\-\d+\-\d+', 'DATE', X)
  
  def replaceYear(X):
    return re.sub(r'[0-2][0-9]{3}', 'YEAR', X)
