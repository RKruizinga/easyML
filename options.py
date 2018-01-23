import argparse
class Options:
  def __init__(self, description):
    self.parser = argparse.ArgumentParser(description=description)
  
  def add(self, name, _type, _default, _help):
    self.parser.add_argument(
      '--'+name, 
      type=_type, 
      default=_default, 
      help=_help)

  def parse(self):
    self.args = self.parser.parse_args()
    self.args_dict = vars(self.args) #args as a dict, for printing purposes