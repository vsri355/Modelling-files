import os
from placentaAnalysisFunctions import *
from placentaAnalysis_utilities import *
from placentagen import imports_and_exports as pg

path = '/hpc/vsri355/Modelling/Updated_rodent_analysis/Mice/BranchList_files-Mice/Con30pt_8' #points to the folder containing the ex nodes and elem files
os.chdir(path)
file_name = "tree"  # file names of nodes and elems
nodes=pg.import_exnode_tree(file_name+'.exnode')
elements=pg.import_exelem_tree(file_name+'.exelem')
node = nodes['nodes']
elem = elements['elems']





