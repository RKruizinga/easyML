import argparse
class Options:
  def __init__(self, description, con):
    self.parser = argparse.ArgumentParser(description=description)

    self.add(name=con.title['name'], _type=con.title['type'], _default=con.title['default'], _help=con.title['help'])
    self.add(name=con.method['name'], _type=con.method['type'], _default=con.method['default'], _help=con.method['help'])
    self.add(name=con.dataset['name'], _type=con.dataset['type'], _default=con.dataset['default'], _help=con.dataset['help'])
    self.add(name=con.random_seed['name'], _type=con.random_seed['type'], _default=con.random_seed['default'], _help=con.random_seed['help'])
    self.add(name=con.avoid_skewness['name'], _type=con.avoid_skewness['type'], _default=con.avoid_skewness['default'], _help=con.avoid_skewness['help'])
    self.add(name=con.KFold['name'], _type=con.KFold['type'], _default=con.KFold['default'], _help=con.KFold['help'])
    self.add(name=con.print_details['name'], _type=con.print_details['type'], _default=con.print_details['default'], _help=con.print_details['help'])
    self.add(name=con.show_fitting['name'], _type=con.show_fitting['type'], _default=con.show_fitting['default'], _help=con.show_fitting['help'])
    
    self.add(name=con.data_folder['name'], _type=con.data_folder['type'], _default=con.data_folder['default'], _help=con.data_folder['help'])
    self.add(name=con.data_method['name'], _type=con.data_method['type'], _default=con.data_method['default'], _help=con.data_method['help'])
    self.add(name=con.predict_label['name'], _type=con.predict_label['type'], _default=con.predict_label['default'], _help=con.predict_label['help'])

  def add(self, name, _type, _default, _help):
    self.parser.add_argument(
      '--'+name, 
      type=_type, 
      default=_default, 
      help=_help)

  def parse(self):
    self.args = self.parser.parse_args()
    self.args_dict = vars(self.args) #args as a dict, for printing purposes