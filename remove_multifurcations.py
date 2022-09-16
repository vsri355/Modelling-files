import os
from placentaAnalysisFunctions import *
from placentaAnalysis_utilities import *
from placentagen import imports_and_exports as pg

path = '/hpc/vsri355/Modelling/Modelling-files/test' #points to the folder containing the ex nodes and elem files
os.chdir(path)
file_name = "trialtree"  # file names of nodes and elems and takes in the node and elem files to be used
nodes=pg.import_exnode_tree(file_name+'.exnode')
elements=pg.import_exelem_tree(file_name+'.exelem')
#initializing nodes and elems
node = nodes['nodes']
elem = elements['elems']

output = 1  # set whether output exfiles are made
if output == 1:
    # cmgui files to be exported
    path = '/hpc/vsri355/Modelling/Modelling-files/CMGUI_files/test'
    os.chdir(path) # filepath to save exported CMGUIfiles
pg.export_ex_coords(node, 'vessels', file_name, 'exnode') # exports nodes
pg.export_exelem_1d(elem, 'vessels', file_name) # exports elems




