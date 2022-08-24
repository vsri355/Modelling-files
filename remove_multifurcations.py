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

output = 1  # set whether output exfiles are made
if output == 1:
    # cmgui files
    path = '//hpc/vsri355/Modelling/Modelling-files/CMGUI_files/Mice/Con30pt_8'
    os.chdir(path)
pg.export_ex_coords(node, 'vessels', file_name, 'exnode')
pg.export_exelem_1d(elem, 'vessels', file_name)
#export_solution_2(radius, 'radii', name, radii)
#export_solution_2(strahler, 'strahler', name+'orders', 'order')



