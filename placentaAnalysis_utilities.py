import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

######
# Contains following functions for placenta analysis:
# sort_elements_order
# check_multiple
# update_elem
# extend_node
# extend_node_subtree
# remove_indexed_row
# remove_rows
# row_swap_2d
# row_swap_1d
# is_member
# plot_vasculature_3d
# find_maximum_joins
# evaluate_orders
# element_connectivity_1D
# export_solution_2
# export_to_ipelem
# export_to_ipnode
# import_ipnode_tree
# import_ipelem_tree
# get_final_integer
# get_final_float
# import_exnode_tree
# import_exelem_tree
# is_float
# find_length_elem
# find_length_elem_single
######

######
# Function: Sorts elements by order descending order
# Inputs: elems - array containing elements where elems=[elem_no, node1, node2]
# Outputs: elems - sorted elements by order
######

def sort_elements_order(elems):
    for ne in range(0,len(elems)):
        for ne2 in range(0,len(elems)):
            if elems[ne][1] < elems[ne2][1]:
                row_swap_2d(elems,ne,ne2)
    if not ne%100:
        print (ne),'out of', len(elems), 'elems'
    return elems

######
# Function: checks whether there are more than 2 elements connected to a node
# Inputs: elem_connect - connectivity information about a set of nodes and elements
# Outputs: max_down - maximum number of downstream elements at a node
######
def check_multiple(elem_connect):
    up = elem_connect['elem_up']
    down = elem_connect['elem_down']
    for i in range(0, len(up)):
        if up[i][0] > 2:
            print ('element ', i, 'has >2 upstream elements')
    max_down=0
    count = 0
    for i in range(0, len(down)):
        if down[i][0] > 2:
            print ('element ', i, 'has ', down[i][0], ' downstream elements')
            if max_down < down[i][0]:
                max_down=down[i][0]
            count = count + 1

    print ('number of elements with >2 down stream: ', count)

    return max_down

######
# Function: creates new element connecting node1 and node2 and updates existing elements that start at node1
# Inputs: elem_i - elem that has more than 2 downstream elements
#		  node2 - new node number
#		  geom - array containing tree information eg. elems, nodes, radii, length, euclidean length
#         elem_connect - array containing element connectivity information of elem_down and elem_up
# Outputs: geom_new - updated input geom array with extra element and node
######
def update_elems(elem_i, node2, geom, elem_connect):
    #unpack inputs
    elem_up = elem_connect['elem_up']
    elem_down = elem_connect['elem_down']
    elems=geom['elems']
    nodes=geom['nodes']

    num_elem = len(elems)
    new_elem = -1 * np.ones(3)
    node1=int(elems[elem_i][2]) #node other end of elem

    # create new elem connecting node1 and new node2
    new_elem[0] = num_elem  # elem numbering starts from 0; [0 1 2] num_elem = 3; new elem = 3
    new_elem[1] = node1
    new_elem[2] = node2

    # add new element to end
    elems = np.vstack((elems, new_elem))

    # update after second downstream element with new node
    for i in range(2,elem_down[elem_i][0]+1):
        old_elem = elem_down[elem_i][i]  # first down stream element
        elems[old_elem][1] = node2  # change starting node of old_elem to new node2

    geom['elems']=elems

    # add copy of node1 geom for node2 at end
    for item in geom.keys():
        current = geom[item]
        # print 'key:', item
        #print 'current', current
        # print 'current[ne]', current[ne]
        if item == 'nodes' or item == 'elems':
            continue #node and element already appended
        elif item == 'length': #radii 1D array
            new_length = find_length_single(nodes, node1, node2)
            current = np.hstack((current, new_length))
        else:
            current = np.hstack((current, current[elem_i]))
        geom[item]=current

    return geom


######
# Function: adds another node to the end of node list that is the input node extended slightly in the longest axis of the
#           associated element
# Inputs: elem_i - index of element associated with >2 downstream elements
#		  nodes - node info in a skeleton where nodes=[x, y, z]
#		  elems - element info where elems=[elem_no, node1, node2] elements and nodes start from 0
# Output: updated nodes - new node appended to end of input nodes
#         node2 - the node number of the new node
######
def extend_node(elem_i, geom):
    nodes=geom['nodes']
    elems=geom['elems']
    num_nodes = len(nodes)
    dif = np.zeros(3)
    new_node = -1 * np.ones(3)

    node1 = int(elems[elem_i][1])
    node2 = int(elems[elem_i][2])  # node at other end of the element
    for i in range(0, 3):
        # assuming nodes starts index = node number (start at 0)
        dif[i] = np.abs(nodes[node1][i] - nodes[node2][i])  # store difference of xyz
    max_i = np.argmax(dif)  # longest axis (x, y or z)
    for i in range(0, 3):
        new_node[i] = nodes[node1][i]  # replicate old node
        if i == max_i:
            if nodes[node2][i] < 0:
                new_node[i] = nodes[node2][i] - 1e-10  # extend node slightly in longest axis
            else:
                new_node[i] = nodes[node2][i] + 1e-10
    # add new node to end
    nodes = np.vstack((nodes, new_node))
    node2 = int(num_nodes)

    return nodes, node2

#subtree input nodes has node numbers in the first column
def extend_node_subtree(elem_i, geom):
    nodes=geom['nodes']
    elems=geom['elems']
    num_nodes = len(nodes)
    dif = np.zeros(3)
    new_node = -1 * np.ones(4)

    node1 = int(elems[elem_i][1])
    node2 = int(elems[elem_i][2])  # node at other end of the element
    for nn in range(0, num_nodes):
        if nodes[nn, 0] == node1:
            nn1 = nn
        elif nodes[nn, 0] == node2:
            nn2 = nn
    for i in range(1, 4):
        dif[i-1] = np.abs(nodes[nn1][i] - nodes[nn2][i])  # store difference of xyz
    max_i = np.argmax(dif) + 1  # longest axis (x, y or z)
    new_node[i] = nodes[nn1][i]  # replicate old node
    for i in range(1, 4):
        new_node[i] = nodes[nn1][i]  # replicate old node
        if i == max_i:
         if nodes[nn2][i] < 0:
            new_node[i] = nodes[nn2][i] - 1e-10  # extend node slightly in longest axis
         else:
            new_node[i] = nodes[nn2][i] + 1e-10
    # add new node to end
    new_node[0]=max(nodes[:,0])+1
    nodes = np.vstack((nodes, new_node))
    node2 = int(new_node[0])

    return nodes, node2


######
# Function: Remove rows from main_array at which an matching index array is equal to index
# Inputs: main_array - an N x M array of values
# 		  index_array - an 1 x N array of values
#		  index - a value that controls removal of rows
# Output: modified main_array with rows removed where corresponding col of index_array == index
######

def remove_indexed_row(main_array, index_array, index):
    i = 0
    while i < len(main_array):
        if index_array[i] == index:
            main_array = np.delete(main_array, i, axis=0)
    i = i + 1
    return main_array


######
# Function: Remove rows from both mainArray and Arrays at which main array has values less than zero
# Inputs: mainArray - an N x M array of values
#         Arrays - a list of arrays each with length N for their first axis
# Outputs: for each row of mainArray for which the first element is below zero; this row is removed from mainArray and from each array
######

def remove_rows(main_array, arrays):
    i = 0

    while i < len(main_array):
        if main_array[i, 0] < 0:  # then get rid of row from all arrays

            for j in range(0, len(arrays)):
                array = arrays[j]
                array = np.delete(array, (i), axis=0)
                arrays[j] = array
            main_array = np.delete(main_array, (i), axis=0)

        else:
            i = i + 1

    return main_array, arrays


######
# Function: Swaps 2 rows in an array
# Inputs: array - a N x M array
#         row1 & row2 - the indices of the two rows to be swapped
# Outputs: array, with row1 and row2 swapped
######

def row_swap_2d(array, row1, row2):
    placeholder = np.copy(array[row1, :])
    array[row1, :] = array[row2, :]
    array[row2, :] = placeholder
    return array


######
# Function: Swaps 2 rows in an array
# Inputs: array - a N x 1 array
#         row1 & row2 - the indices of the two rows to be swapped
# Outputs: array, with row1 and row2 swapped
######

def row_swap_1d(array, row1, row2):
    placeholder = np.copy(array[row1])
    array[row1] = array[row2]
    array[row2] = placeholder
    return array


######
# Function: Finds first occurrence of a specified row of values in an array or returns -1 if the given row is not present
#           Similar to Matlab isMember function
# Inputs: matrix - an N x M array
#         v - a 1 x M array
# Outputs: index at which v first occurs in matrix, or else -1
######

def is_member(v, matrix):
    L = (np.shape(matrix))
    L = L[0]

    for i in range(0, L):
        if np.array_equal(v, matrix[i, :]):
            index = i
            return index
    return -1


######
# Function: Creates a 3D plot of branching tree
# Inputs: nodes - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#         elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         colour - an N x 1 array where value determines colour of corresponding element
#         Nc - the maximum number of elements connected at a single node
# Outputs: 3D plot of tree, with radius proportional to radii and colour depending on the input array
######

def plot_vasculature_3d(nodes, elems, colour, radii):
    # initialize arrays
    Ne = len(elems)
    elems = elems[:, 1:3]
    x = np.zeros([Ne, 2])
    y = np.zeros([Ne, 2])
    z = np.zeros([Ne, 2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # scale colour and radii
    colour = (colour - min(colour)) / max(colour) * 255
    radii = radii / max(radii) * 3

    for i in range(0, Ne):
        # get start and end node
        nN1 = int(elems[i, 0])
        nN2 = int(elems[i, 1])

        # get coordinates of nodes
        x[i, 0] = nodes[nN1, 0]
        y[i, 0] = nodes[nN1, 1]
        z[i, 0] = nodes[nN1, 2]
        x[i, 1] = nodes[nN2, 0]
        y[i, 1] = nodes[nN2, 1]
        z[i, 1] = nodes[nN2, 2]

        colour_value = np.asarray(cm.jet(int(colour[i])))
        ax.plot(np.squeeze(x[i, :]), np.squeeze(y[i, :]), np.squeeze(z[i, :]), c=colour_value[0:3], linewidth=radii[i])

    plt.show()

    return 0


######
# Function: Finds the maximum number of elements that join at one node
# Inputs: elems - an N x 3 array containing element number in the first column and node indices in the second two columns
# Outputs: Nc - the maximum number of elements that join at one node
######

def find_maximum_joins(elems):
    elems = np.concatenate([np.squeeze(elems[:, 1]), np.squeeze(elems[:, 2])])
    elems = elems.astype(int)
    result = np.bincount(elems)
    Nc = (max(result)) + 1

    # Warning if detect an unusual value
    if Nc > 10:
        print('Warning, large number of elements at one node: ' + str(Nc))
        Nc = 10

    return Nc


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Note: only works for diverging trees
# Modifications ensure that function works for more than three elements joining at one node
# Inputs: elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         elem_connect - connectivity of elements, as created by element_connectivity_1D
# Outputs: orders, containing 3 N x 1 arrays which give the Strahler / Horsefield / Generation of each element
######

def evaluate_orders(elems, elem_connect):
    num_elems = len(elems)

    elem_upstream = elem_connect['elem_up']
    elem_downstream = elem_connect['elem_down']

    # Initialise order definition arrays
    strahler = np.zeros(len(elems), dtype=int)
    horsfield = np.zeros(len(elems), dtype=int)
    generation = np.zeros(len(elems), dtype=int)

    # Calculate generation of each element
    maxgen = 1  # Maximum possible generation
    for ne in range(0, num_elems):
        ne0 = elem_upstream[ne][1]
        if ne0 != 0:
            # Calculate parent generation
            n_generation = generation[ne0]
            if elem_downstream[ne0][0] == 1:
                # Continuation of previous element
                generation[ne] = n_generation
            elif elem_downstream[ne0][0] >= 2:
                # Bifurcation (or morefurcation)
                generation[ne] = n_generation + 1
        else:
            generation[ne] = 1  # Inlet
        maxgen = np.maximum(maxgen, generation[ne])

    # Now need to loop backwards to do ordering systems
    for ne in range(num_elems - 1, -1, -1):
        n_horsfield = np.maximum(horsfield[ne], 1)
        n_children = elem_downstream[ne][0]
        if n_children == 1:
            if generation[elem_downstream[ne][1]] == 0:
                n_children = 0
        temp_strahler = 0
        strahler_add = 1
        if n_children >= 2:  # Bifurcation downstream
            temp_strahler = strahler[elem_downstream[ne][1]]  # first daughter
            for noelem in range(1, n_children + 1):
                ne2 = elem_downstream[ne][noelem]
                temp_horsfield = horsfield[ne2]
                if temp_horsfield > n_horsfield:
                    n_horsfield = temp_horsfield
                if strahler[ne2] < temp_strahler:
                    strahler_add = 0
                elif strahler[ne2] > temp_strahler:
                    strahler_add = 0
                    temp_strahler = strahler[ne2]  # strahler of highest daughter
            n_horsfield = n_horsfield + 1
        elif n_children == 1:
            ne2 = elem_downstream[ne][1]  # element no of daughter
            n_horsfield = horsfield[ne2]
            strahler_add = strahler[ne2]
        horsfield[ne] = n_horsfield
        strahler[ne] = temp_strahler + strahler_add

    return {'strahler': strahler, 'horsfield': horsfield, 'generation': generation}


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Modifications ensure that function works for more than three elements joining at one node
# Inputs: node_loc - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#         elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         Nc - the maximum number of elements connected at a single node
# Outputs: elem_up: an N x Nc array containing indices of upstream elements (the first value is number of upstream elements)
#          elem_down: an N x Nc array containing indices of downstream elements (the first value is number of downstream elements)
######

def element_connectivity_1D(node_loc, elems, Nc):
    # Initialise connectivity arrays
    num_elems = len(elems)
    elem_upstream = np.zeros((num_elems, Nc), dtype=int)
    elem_downstream = np.zeros((num_elems, Nc), dtype=int)

    num_nodes = len(node_loc)
    elems_at_node = np.zeros((num_nodes, Nc), dtype=int)

    # determine elements that are associated with each node
    for ne in range(0, num_elems):
        for nn in range(1, 3):
            nnod = int(elems[ne][nn])
            elems_at_node[nnod][0] = elems_at_node[nnod][0] + 1
            elems_at_node[nnod][elems_at_node[nnod][0]] = ne

    # assign connectivity
    for ne in range(0, num_elems):
        nnod2 = int(elems[ne][2]) #end node of elem2

        for noelem in range(1, elems_at_node[nnod2][0] + 1):
            ne2 = elems_at_node[nnod2][noelem] #other elements at end node

            if ne2 != ne: #not upstream elem
                elem_upstream[ne2][0] = elem_upstream[ne2][0] + 1
                elem_upstream[ne2][elem_upstream[ne2][0]] = ne
                elem_downstream[ne][0] = elem_downstream[ne][0] + 1
                elem_downstream[ne][elem_downstream[ne][0]] = ne2

    return {'elem_up': elem_upstream, 'elem_down': elem_downstream}


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Writes values to a cmgui exelem file
# Inputs: data - an N x 1 array with a value for each element in the tree
#         groupname - group name that will appear in cmgui
#         filename - name that the file is saved as
#         name - name that values will be called in cmgui
# Outputs: an "exelem" file containing the data value for each element, named according to names specified
######

def export_solution_2(data, groupname, filename, name):
    # Write header
    type = "exelem"
    data_num = len(data)
    filename = filename + '.' + type
    f = open(filename, 'w')
    f.write(" Group name: %s\n" % groupname)
    f.write("Shape. Dimension=1\n")
    f.write("#Scale factor sets=0\n")
    f.write("#Nodes=0\n")
    f.write(" #Fields=1\n")
    f.write("1) " + name + ", field, rectangular cartesian, #Components=1\n")
    f.write(name + ".  l.Lagrange, no modify, grid based.\n")
    f.write(" #xi1=1\n")

    # Write element values
    for x in range(0, data_num):
        f.write(" Element:            %s 0 0\n" % int(x + 1))
        f.write("   Values:\n")
        f.write("          %s" % np.squeeze(data[x]))
        f.write("   %s \n" % np.squeeze(data[x]))
    f.close()

    return 0


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Writes values to a cmgui ipelem file
# Inputs: data - an N x 3 array with a element information for the tree =[elem no., node1, node2]
#         name - name in heading in cmgui
#         filename - name that the file is saved as
# Outputs: an "ipelem" file containing the data value for each element, named according to names specified
######

def export_to_ipelem(data, name, filename):
    # Write header
    type = "ipelem"
    data_num = len(data)
    filename = filename + '.' + type
    f = open(filename, 'w')
    f.write(" CMISS Version 2.1  ipelem File Version 2\n")
    f.write(" Heading: %s\n\n" % name)
    f.write(" The number of elements is [1]: %s \n\n" % int(data_num))

    # Write element values
    for x in range(0, data_num):
        f.write(" Element number [    1]:     %s\n" % int(x + 1))
        f.write(" The number of geometric Xj-coordinates is [3]: 3\n")
        f.write(" The basis function type for geometric variable 1 is [1]:  1\n")
        f.write(" The basis function type for geometric variable 2 is [1]:  1\n")
        f.write(" The basis function type for geometric variable 3 is [1]:  1\n")
        f.write(" Enter the 2 global numbers for basis 1: %s %s\n\n" % (int(data[x][1] + 1), int(data[x][2] + 1)))

    f.close()

    return 0


######
# Modified from placentagen (https://github.com/VirtualPregnancy/placentagen)
# Writes values to a cmgui ipelem file
# Inputs: data - an N x 3 array with a element information for the tree =[elem no., node1, node2]
#         name - name in heading in cmgui
#         filename - name that the file is saved as
# Outputs: an "ipelem" file containing the data value for each element, named according to names specified
######

def export_to_ipnode(data, name, filename):
    # Write header
    type = "ipnode"
    data_num = len(data)
    filename = filename + '.' + type
    f = open(filename, 'w')
    f.write(" CMISS Version 2.1  ipnode File Version 2\n")
    f.write(" Heading: %s\n\n" % name)
    f.write(" The number of nodes is [1]: %s \n" % int(data_num))
    f.write(" Number of coordinates [3]: 3\n")
    f.write(" Do you want prompting for different versions of nj=1 [N]? N\n")
    f.write(" Do you want prompting for different versions of nj=2 [N]? N\n")
    f.write(" Do you want prompting for different versions of nj=3 [N]? N\n")
    f.write(" The number of derivatives for coordinate 1 is [0]: 0\n")
    f.write(" The number of derivatives for coordinate 2 is [0]: 0\n")
    f.write(" The number of derivatives for coordinate 3 is [0]: 0\n")

    # Write element values
    for x in range(0, data_num):
        f.write(" Node number [    1]:     %s\n" % int(x + 1))
        f.write(" The Xj(1) coordinate is [ 0.00000E+00]:  %s\n" % data[x][0])
        f.write(" The Xj(2) coordinate is [ 0.00000E+00]:  %s\n" % data[x][1])
        f.write(" The Xj(3) coordinate is [ 0.00000E+00]:  %s\n\n" % data[x][2])

    f.close()

    return 0

######
# Function: Takes an ipelem file from CMGUI and returns an array of elements and the total number of elements
# Inputs:	ipelem file containing element information for a tree
# Outputs:	total_elems - total number of elements
#			elems - an N x 3 array, where elems=[element_number, start_node, end_node] for N elements
######

def import_ipelem_tree(filename):
    # count element for check of correct number for the user, plus use in future arrays
    count_el = 0
    total_el = 0
    # Initialise array of el numbers and values
    el_array = np.empty((0,3),dtype = int)
    # open file
    with open(filename) as f:
        # loop through lines of file
        while 1:
            line = f.readline()
            if not line:
                break  # exit if done with all lines
            # identifying whether there is an element defined here
            if line.find("Element number") != -1: #line defines new element
            	count_el = count_el + 1  # count the element
                count_atribute = 0  # intitalise attributes of the element (1st node, 2nd node)
                el_array = np.append(el_array, np.zeros((1, 3),dtype = int), axis=0)
                el_array[count_el - 1][count_atribute] = get_final_integer(line) - 1 #element number in first col
            else:
                if line.find("global") != -1: #if right line
                	count_atribute = count_atribute + 1
                	ibeg=line.index(":")
                	iend=len(line)
                	sub_string=line[ibeg+1:iend]
                	sub_string=sub_string.strip()
                	imid=sub_string.index(" ")
                	el_array[count_el - 1][count_atribute] = float(sub_string[0:imid])  # 1st node of element
                	el_array[count_el - 1][count_atribute + 1] = float(sub_string[imid+1:len(sub_string)])  # 2nd node of element

    total_el = count_el
    return total_el, el_array

######
# Function: Takes an ipnode file from CMGUI and returns an array of nodes and the total number of nodes
# Inputs:	ipnode file containing node information for a tree
# Outputs:	total_nodes - total number of nodes
#			nodes - an M x 3 array where nodes=[x_coord, y_coord, z_coord] for M nodes
######

def import_ipnode_tree(filename):
    # count nodes for check of correct number for the user, plus use in future arrays
    count_node = 0
    # Initialise array of node numbers and values
    node_array = np.empty((0,3), dtype=float)
    # open file
    with open(filename) as f:
        # loop through lines of file
        while 1:
            line = f.readline()
            if not line:
                break  # exit if done with all lines
            # identifying whether there is a node defined here
            if line.find("Node number") != -1: #line defines new node
            	count_node = count_node + 1  # count the node
                count_atribute = 0  # intitalise attributes of the node (coordinates, radius)
                node_array=np.append(node_array,np.zeros((1,3)),axis = 0)  # initialise a list of attributes for each node
                node_array[count_node - 1][count_atribute] = get_final_integer(line)
            else:
            	ibeg = line.find("Xj(")
                if  ibeg != -1:
                	count_atribute = int(line[ibeg+3]) - 1
                	node_array[count_node - 1][count_atribute] = get_final_float(line)
    total_nodes = count_node
    return total_nodes,node_array

######
# Finds the integer at the end of a string
# Inputs: string
# Outputs: integer
######

def get_final_integer(string):
	iend=len(string) #get the length of the string
	ntemp=string.split(":")[1] #get characters after ":" in string
	if ntemp[0] =='-': #check for negative sign
		nsign=-1
	else:
		nsign=1
	num = int(ntemp)*nsign #apply sign to int
	return num

######
# Finds the float at the end of a string
# Inputs: string
# Outputs: float
######

def get_final_float(string):
	iend=len(string) #get the length of the string
	ntemp=string.split(":")[1] #get characters after ":" in string
	if ntemp[0] =='-': #check for negative sign
		nsign=-1
	else:
		nsign=1
	num = float(ntemp)*nsign #apply sign to real
	return num


######
# Function: Takes an exnode file from CMGUI and returns an array of nodes and the total number of nodes
# Inputs:	exnode file containing node information for a tree
# Outputs:	total_nodes - total number of nodes
#			nodes - an M x 3 array where nodes=[x_coord, y_coord, z_coord] for M nodes
######

def import_exnode_tree(filename):
    # count nodes for check of correct number for the user, plus use in future arrays
    count_node = 0
    # Initialise array of node numbers and values
    node_array = np.empty((0, 4))
    # open file
    with open(filename) as f:
        # loop through lines of file
        while 1:
            line = f.readline()
            if not line:
                break  # exit if done with all lines
            # identifying whether there is a node defined here
            line_type = str.split(line)[0]
            if (line_type == 'Node:'):  # line defines new node
                count_node = count_node + 1  # count the node
                count_atribute = 0  # intitalise attributes of the node (coordinates)
                node_array = np.append(node_array, np.zeros((1, 4)),
                                       axis=0)  # initialise a list of attributes for each node
                node_array[count_node - 1][count_atribute] = int(str.split(line)[1]) - 1
            else:
                line_num = is_float(line_type)  # checking if the line is a number
                if (line_num):  # it is a number
                    if not "index" in line:
                        count_atribute = count_atribute + 1
                        node_array[count_node - 1][count_atribute] = float(str.split(line)[0])
    total_nodes = count_node
    return {'total_nodes': total_nodes, 'nodes': node_array}


######
# Function: Takes an exelem file from CMGUI and returns an array of elements and the total number of elements
# Inputs:	exelem file containing element information for a tree
# Outputs:	total_elems - total number of elements
#			elems - an N x 3 array, where elems=[element_number, start_node, end_node] for N elements
######

def import_exelem_tree(filename):
    # count element for check of correct number for the user, plus use in future arrays
    count_el = 0
    # Initialise array of el numbers and values
    el_array = np.empty((0, 3), dtype=int)
    # open file
    with open(filename) as f:
        # loop through lines of file
        while 1:
            line = f.readline()
            if not line:
                break  # exit if done with all lines
            # identifying whether there is an element defined here
            line_type = str.split(line)[0]

            if (line_type == 'Element:'):  # line defines new el
                count_el = count_el + 1  # count the el
                count_atribute = 0  # intitalise attributes of the el (1st el, 2nd el)
                el_array = np.append(el_array, np.zeros((1, 3), dtype=int), axis=0)
                el_array[count_el - 1][count_atribute] = int(str.split(line)[1]) - 1
            elif (line_type == 'Nodes:'):  # checking if the line is a node
                count_atribute = count_atribute + 1
                el_array[count_el - 1][count_atribute] = float(str.split(line)[1]) - 1  # first node of element
                el_array[count_el - 1][count_atribute + 1] = float(str.split(line)[2]) - 1  # 2nd node of element

    total_el = count_el
    return {'total_elems': total_el, 'elems': el_array}

def is_float(str):
    try:
        num = float(str)
    except ValueError:
        return False
    return True


def find_length_elem(elem, nodes):
    distance = -1 * np.ones(len(elem))

    # go through all elems
    for ne in range(0, len(elem)):
        node1 = elem[ne][1]
        node2 = elem[ne][2]
        d_x = nodes[node1][1] - nodes[node2][1]
        d_y = nodes[node1][2] - nodes[node2][2]
        d_z = nodes[node1][3] - nodes[node2][3]
        distance[ne] = np.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)

    return distance

def find_length_single(nodes,node1,node2):

    d_x = nodes[node1][0] - nodes[node2][0]
    d_y = nodes[node1][1] - nodes[node2][1]
    d_z = nodes[node1][2] - nodes[node2][2]
    distance = np.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)

    return distance

