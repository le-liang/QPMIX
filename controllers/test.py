import os
from os.path import dirname
current_dir = dirname(dirname(os.path.realpath(__file__)))
print(current_dir)