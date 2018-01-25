class Constants:
  title = {
    'name': 'title',
    'type': str,
    'default': 'Classification',
    'help': 'Title of the system'
  }

  method = {
    'name': 'method',
    'type': str,
    'default': 'svm',
    'help': 'Machine Learning Technique'
  }

  random_seed = {
    'name': 'random_seed',
    'type': int,
    'default': 3,
    'help': 'Random Seed number'
  }

  predict_label = {
    'name': 'predict_label',
    'type': str,
    'default': 'emoji',
    'help': 'Which label to predict'
  }

  avoid_skewness = {
    'name': 'avoid_skewness',
    'type': bool,
    'default': False,
    'help': 'Should we make the data unskewed?'
  }

  KFold = {
    'name': 'k',
    'type': int,
    'default': 1,
    'help': 'Cross validation K'
  }

  dataset = {
    'name': 'dataset',
    'type': list,
    'default': ['train', 'development', 'test'],
    'help': 'Specify which datasets you want to use ([train, development, test])'
  }

  data_folder = {
    'name': 'data_folder',
    'type': str,
    'default': 'data/emoji_prediction/',
    'help': 'Specify in which folder the files are)'
  }

  data_method = {
    'name': 'data_method',
    'type': int,
    'default': 1,
    'help': 'How to read the data'
  }

  def __init__(self):
    pass