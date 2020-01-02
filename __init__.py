"""
Non-negative matrix and tensor factorization init file

Imports all modules in subfolder modules

"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19

import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],'modules')))
if cmd_subfolder not in sys.path:
   sys.path.insert(0, cmd_subfolder)

from nmtf.modules.nmtf import *