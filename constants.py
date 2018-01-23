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

  data_method = {
    'name': 'data_method',
    'type': int,
    'default': 1,
    'help': 'How to read the data'
  }

  predict_languages = {
    'name': 'predict_languages',
    'type': str,
    'default': 'e',
    'help': 'Which languages to predict'
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
    'name': 'KFold',
    'type': int,
    'default': 1,
    'help': 'Cross validation K'
  }


  def __init__(self):
    pass