print("Good luck!")

from .basic import *
from .dmaps import *
from .loader import *
from .plotting import *
from .Units import *

import sys
import os
"""
DEFINE ABSOLUTE PATH TO ../CUSTOM/
"""
path_to_custom = "/home/mt/Penny/Custom/Custom.py"
try: exec(open(path_to_custom).read())
except FileNotFoundError:
    print("Bad definition of Path_to_custom\nDefine in Penny/Penny/__init__.py and reinstall")

