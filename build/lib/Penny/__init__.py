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
exec(open(path_to_custom).read())
#path_Custom = "../"
#os.popen("ls -d "+path_Custom+"Custom.py").read()
#sys.path.append(os.popen("readlink -f ../Custom").read().split()[0]+"" )
#import Custom
#path_custom = os.popen("readlink -f ../Custom/Custom.py").read().split()[0]
#exec(open(path_custom).read())
