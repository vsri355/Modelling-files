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

#for removing multiple connections in the read in tree these were the lines kejia have used
seed_geom=seed_geometry_rodent(volume,thickness,ellipticity,cord_insertion_x,cord_insertion_y,umb_artery_length,datapoints_chorion) #this line should be modified to understand the tree
elem_connect=element_connectivity_1D(seed_geom['nodes'],seed_geom['elems'],7)

#the variables passed in this remove multiple connections function should refer to the read in tree's
seed_geom,elem_connect=remove_multiple_elements(seed_geom,elem_connect,'non')
#after this step again the nodes and elems should be updated and assigned with names.
node=seed_geom['nodes']
seed_geom['nodes']=node[:,1:]

#something with return() should be here which returns the final tree after removing the multiple connections

output = 1  # set whether output exfiles are made
if output == 1:
    # cmgui files to be exported
    path = '/hpc/vsri355/Modelling/Modelling-files/CMGUI_files/test'
    os.chdir(path) # filepath to save exported CMGUIfiles
pg.export_ex_coords(node, 'vessels', file_name, 'exnode') # exports nodes
pg.export_exelem_1d(elem, 'vessels', file_name) # exports elems


