"""
Non-negative Matrix and Tensor Factorization

Author: Paul Fogel

License: MIT

Release date: Jan 6 '20

Version: 11.0.0

https://github.com/paulfogel/NMTF

"""

from .modules import *
from . import *
import os
import sys
import inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], 'modules')))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
