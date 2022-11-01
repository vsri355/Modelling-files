import os
import placentagen as pg


file_name = "test-inputs/trialtree"  # file names of nodes and elems and takes in the node and elem files to be used
#for removing multiple connections in the read in tree
chorion_and_stem = {}  #initializing the tree geometry
chorion_and_stem['nodes'] = pg.import_exnode_tree(file_name+'.exnode')['nodes']
chorion_and_stem['elems'] = pg.import_exelem_tree(file_name+'.exelem')['elems']

#removing multiple connections
chorion_and_stem=pg.remove_multiple_elements(chorion_and_stem['nodes'],chorion_and_stem['elems'])


# cmgui files to be exported
file_name = 'CMGUI_files-output/test/trialtree'
pg.export_ex_coords(chorion_and_stem['nodes'], 'vessels', file_name, 'exnode') # exports nodes
pg.export_exelem_1d(chorion_and_stem['elems'], 'vessels', file_name) # exports elems




