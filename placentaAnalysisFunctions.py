import numpy as np
from tabulate import tabulate
from placentaAnalysis_utilities_new import *

######
#Contains following functions for placenta analysis:
#remove_multiple_elements
#simplify_tree
#remove_unconnected_nodes
#sort_data
#sort_elements
#arrange_by_strahler_order
#prune_by_order
#find_branch_angles
#summary_statistics
######

######
# Function: Removes elements with more than 2 downstream elements by adding a new element of minimal length and
#           reallocating down stream elements to this
# Inputs: geom - structure containing node and element information, radii, length etc.
#         elem_connect - structure containing information for up and downstream elements for each elem
# Outputs: geom - updated structure with elements having a maximum of 2 downstream elements
######

def remove_multiple_elements(geom, elem_connect, type):
    # unpackage information
    max_down = check_multiple(elem_connect)

    while max_down > 2:
        elem_down = elem_connect['elem_down']
        for ne in range(0, len(elem_down)):
            if elem_down[ne][0] > 2:  # more than 2 connected downstream
                if type == 'subtree':
                    geom['nodes'], node2 = extend_node_subtree(ne, geom)  # create new node
                    geom = update_elems(ne, node2, geom, elem_connect)  # create new element and update old
                else:
                    geom['nodes'], node2 = extend_node(ne, geom)  # create new node
                    geom = update_elems(ne, node2, geom, elem_connect)  # create new element and update old
        if type == 'subtree':
            elem_connect = element_connectivity_1D_subtree(geom['nodes'], geom['elems'], 6)
        else:
            elem_connect = element_connectivity_1D(geom['nodes'], geom['elems'], 6)

        max_down = check_multiple(elem_connect)
    num_elems = len(geom['elems'])
    elem_down = elem_connect['elem_down']
    elem_up = elem_connect['elem_up']
    geom['elem_down'] = elem_down[:,0:3]
    geom['elem_up'] = elem_up[:,0:3]


    return geom, elem_connect



######
# Function: Simplifies vascular tree to specified number of strahler orders
# Inputs:	n - number of strahler orders to keep (higher orders are kept first)
#			geom - contains elems, nodes and other properties of the skeleton
#			Nc - maximum number of elements connected to a node
# Outputs:	
######

def simplify_tree(n, geom, orders):
	strahler=orders['strahler']
	max_order=np.max(strahler)
	index_orders=np.zeros(len(orders['strahler']))
	
	#remove orders starting from 1 to reach no. specified remaining
	for i in range(1,max_order-n+1):
		geom['elems'] = remove_indexed_row(geom['elems'], orders['strahler'],i)
		geom['radii'] = remove_indexed_row(geom['radii'], orders['strahler'],i)
		geom['length'] = remove_indexed_row(geom['length'], orders['strahler'],i)
		geom['euclidean length'] = remove_indexed_row(geom['euclidean length'], orders['strahler'],i)
		geom['branch_angles'] = remove_indexed_row(geom['branch_angles'], orders['strahler'],i)
		geom['diam_ratio'] = remove_indexed_row(geom['diam_ratio'], orders['strahler'],i)
		geom['length_ratio'] = remove_indexed_row(geom['length_ratio'], orders['strahler'],i)
		
		for j in range(0, len(orders['strahler'])):
			if strahler[j]==i:
				index_orders[j]=i
		print (index_orders)
		orders['strahler']=remove_indexed_row(orders['strahler'],index_orders,i)
		orders['generation']=remove_indexed_row(orders['generation'],index_orders,i)
	
	#remove unconnected nodes from elements removed
	Nc=find_maximum_joins(geom['elems'])
	geom['nodes'],geom['elems']=remove_unconnected_nodes(geom['nodes'], geom['elems'], Nc)
	
	return geom, orders


######
# Function: Removes redundant nodes not connected to any element
# Inputs:	node_loc - N x 3 array, with the (x, y, z) coordinates of node
#			elems - an N x 3 array, the first column in the element number, the second two columns are the index of the start and end node
#			Nc - maximum number of elements connected to a node
# Outputs:	modified elems and node_loc arrays with redundant nodes removed and renumbered in elems
######

def remove_unconnected_nodes(nodes, elems):
    print ('removing......')
    num_elems = len(elems)
    num_nodes = len(nodes)
    redundant_i = []
    # create copy as elements modified
    new_elems = elems.copy()

    print ('no elements:'), num_elems
    print ('no nodes:'), num_nodes

    # find redundant node indexes
    for nn in range(1, num_nodes):  # go through all rows
        (rows, cols) = np.where(elems[:, 1:] == nn)  # find associated elements with nodes
        count = len(cols)
        # print 'nn:', nn,'row_i:',places[0],'col_i:',places[1], 'len:', size
        if count == 0:
            redundant_i = np.append(redundant_i, nn)

    print (len(redundant_i)), 'redundant nodes found:', redundant_i

    # delete redundant nodes
    new_nodes = np.delete(nodes, redundant_i, axis=0)

    if (len(nodes[0])) == 3: #if 3 then update node numbers, else node numbers unchanged as given in nodes[:,0]
        # demote node number if after a removed node
        for i in range(0, len(redundant_i)):  # go through each redundant node
            node = int(redundant_i[i])
            for ne in range(0, num_elems):
                n1 = elems[ne, 1]
                n2 = elems[ne, 2]
                if n1 > node:
                    new_elems[ne, 1] = new_elems[ne, 1] - 1
                if n2 > node:
                    new_elems[ne, 2] = new_elems[ne, 2] - 1

    return new_nodes, new_elems

######
# Function: takes data from the csv and converts it to arrays
# Inputs: data_file - generated from the panadas read_csv function, containing results from imageJ image analysis
#         Arrays - a group of arrays each with length N for their first axis
# Outputs: nodes - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#         elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#         radii, length, euclidean_length - there are all an N x 1 array containing a property for each element
######

def sort_data(data_file):

    # get rid of any skeletons other than the main one
    data_file=data_file[data_file.SkeletonID == 1]

    # get skeleton properties as arrays
    euclid_length=data_file.Euclideandistance.values
    length = data_file.Branchlength.values
    radii = data_file.averageintensityinner3rd.values

    # get elem and node data
    data_file=data_file.drop(['SkeletonID','Branchlength','averageintensityinner3rd','Euclideandistance'], axis=1)
    data_file=data_file.values
    (elems, nodes) = sort_elements(data_file[:, 0:3],data_file[:,3:6])

    # get rid of dud elements
    (elems, [length, euclid_length, radii])=remove_rows(elems, [length, euclid_length, radii])

    return {'nodes': nodes, 'elems':elems, 'radii': radii, 'length': length, 'euclidean length': euclid_length}


######
# Function: takes a list of node pairs (v1, v2) and creates a list of nodes and elements
# Inputs: v1 - N x 3 array of start node coordinates
#         v2 - N x 3 array of end node coordinates
# Outputs: nodes - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#          elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#          elements that start and end at the same node are given a value of -1
######

def sort_elements(v1, v2):

    Nelem=len(v1)
    elems = np.zeros([Nelem, 3])
    nodes = np.zeros([Nelem*2, 3]) # max number of nodes possible

    iN=0 # node index

    # go through first node list
    for iE in range(0, Nelem):

        v=v1[iE,:]
        index = is_member(v, nodes[0:iN][:]) # see if the node is in the nodes list

        if index == -1: # if not, create a new node
            nodes[iN, :]= v
            index=iN
            iN=iN+1

        # else just use index of existing node
        elems[iE,1] = int(index)
        elems[iE, 0] = int(iE) # first column of elements is just the element number

    # go through second node list
    for iE in range(0, Nelem ):

        v = v2[iE, :]
        index = is_member(v, nodes[0:iN, :])

        if index == -1:
            nodes[iN, :] = v
            index = iN
            iN = iN + 1

        elems[iE, 2] = int(index)

        if elems[iE][1]==elems[iE][2]:
            elems[iE,0:2]=int(-1)

    nodes = nodes[0:iN:1][:] # truncate based on how many nodes were actually assigned
    return (elems, nodes)


######
# Function: rearranges elems (and corresponding properties) according to their strahler order, to be compatible with placentagen functions
# Inputs: geom - contains elems, nodes and other properties of the skeleton
#         inlet_loc - the coordinates of the parent node for the entire tree
# Outputs: geom - contains elems and properties, reordered according to strahler order so that no element can be higher in the element list than a higher order branch
######

def arrange_by_strahler_order(geom, inlet_loc):

    # set up arrays
    nodes=geom['nodes']
    elem_properties=np.column_stack([geom['radii'],geom['length'], geom['euclidean length'], geom['elems']])
    elems = np.copy(geom['elems'])  # as elems is altered in this function
    elems=elems[:,1:3] #get rid of first column which means nothing

    Ne = len(elems)
    Nn = len(nodes)
    elem_properties_new = np.zeros([Ne,6])

    # find root node and element from its coordinates
    Nn_root=is_member(inlet_loc, nodes)
    if (Nn_root==-1):
        print("Warning, root node not located")

    #find root element
    Ne_place=np.where(elems==Nn_root) # Ne_place = [row,col] indexes of Nn_root (index=elem-1)
    Ne_root = Ne_place[0]  # only need first index
    if len(Ne_root) > 1:
        print("Warning, root node is associated with multiple elements")
    if len(Ne_root) == 0:
        print("Warning, no root element located")
    Ne_root = Ne_root[0]
    # make root element the first element
    elems=row_swap_2d(elems, 0, Ne_root) #root is elems[0, :]
    elem_properties=row_swap_2d(elem_properties, 0, Ne_root)

    # get element pointing right way
    if (np.squeeze(Ne_place[1])!= 1):
        elems[0,:]=row_swap_1d(np.squeeze(elems[0,:]),1,0)
        elem_properties[0,4:6] = row_swap_1d(np.squeeze(elem_properties[0,4:6]),1,0)

    # find orders
    counter=1
    counter_new=0
    while (counter<Ne):
        # find elements which are terminal
        terminal_elems = np.zeros([Ne, 1])

        # go through each node
        for i in range(0, Nn+1):

            # find number of occurrences of the node
            places = np.where(elems == i)
            ind1 = places[0] #array of rows index
            ind2 = places[1] #array of col index

            if (len(ind1) == 1) and ((ind1[0]) != 0): #if occurs once, then element is terminal (avoids root element)

                ind1 = ind1[0] #first occurrence row index
                ind2 = ind2[0] #first occurrence col index

                # swap to ensure element points right way
                if ind2==0:
                    elems[ind1,:]=row_swap_1d(np.squeeze(elems[ind1,:]),1,0)
                    elem_properties[ind1, 4:6] = row_swap_1d(np.squeeze(elem_properties[ind1,4:6]), 1, 0)

                # assign element under the new element ordering scheme
                elem_properties_new[counter_new,:]=elem_properties[ind1,:]
                counter_new=counter_new+1

                terminal_elems[ind1] = 1

                # join up element with upstream elements
                
                nodeNumNew = elems[ind1, 0] #this is node number at other end of element
                nodeNum=i
                places = np.where(elems == nodeNumNew)  # find where the new node occurs
                
                ind1 = places[0]
                ind2 = places[1]
                
                counter2 = 1
                while ((len(ind1) == 2) & (counter2 < Ne)):  # as can only be present twice if a joining node

                    # see if branch joins to yet another branch, that we haven't yet encountered (i.e. not nodeNum)
                    if (elems[ind1[0], ~ind2[0]] == nodeNum):
                        k = 1
                    else:
                        k = 0
                    terminal_elems[ind1[k]] = 1 # label terminal_elems as joining elements

                    # switch the way element points
                    if (ind2[k] == 0):
                        elems[ind1[k], :] = row_swap_1d(np.squeeze(elems[ind1[k], :]), 1, 0)
                        elem_properties[ind1[k], 4:6] = row_swap_1d(np.squeeze(elem_properties[ind1[k],4:6]), 1, 0)

                    nodeNum = nodeNumNew
                    nodeNumNew = elems[ind1[k], 0]

                    #assign new order
                    elem_properties_new[counter_new, :] = elem_properties[ind1[k], :]
                    counter_new = counter_new + 1

                    # update loop criteria
                    places = np.where(elems == nodeNumNew)
                    ind1 = places[0]
                    ind2 = places[1]
                    counter2 = counter2 + 1

        # update elems to 'get rid of' terminal elements from the list
        terminal_elems[0]= 0 #the root node can never be terminal
        terminal_elems_pair=np.column_stack([terminal_elems, terminal_elems])
        elems[terminal_elems_pair == 1] = -1

        # loop exit criteria
        places=np.where(terminal_elems == 1)
        places=places[1]
        if len(places)==0:
            counter = Ne+1
        counter = counter + 1

    # check for error
    if np.sum(elem_properties_new[Ne-2,:])==0:
        print('Warning, not all elements assigned to new order')

    # assign root element in new order systems
    elem_properties_new[Ne-1, :] = elem_properties[0, :]

    # reverse order
    elem_properties_new= np.flip(elem_properties_new,0)

    elems = geom['elems']
    elems[:,1:3]=elem_properties_new[:,4:6]

    return {'elems': elems, 'radii': elem_properties_new[:,0],'length': elem_properties_new[:,1], 'euclidean length': elem_properties_new[:,2], 'nodes': nodes}


######
# Function: removed terminal branches that connect directly to high order branches
# Inputs: geom - contains elems, and various element properties (length, radius etc.)
#         orders - contains strahler order of each element orders
#         threshold_order - the maximum order which a terminal element can be directly connected to
# Outputs: elems and their corresponding properties are truncated to remove the rows containing terminal elements that connect directly to high order branches
######

def prune_by_order(geom, orders, threshold_order):

     # define arrays
     elem_properties = [geom['length'], geom['euclidean length'], geom['radii']]
     elems=geom['elems']
     elems=elems[:,1:3]

     terminalList = np.where(orders == 1)
     terminalList=terminalList[0]

     # go through list of terminal elements
     for i in range(0, len(terminalList)):
         row = terminalList[i]

         # find parents at the non terminal end of the element, and their order
         ind=np.where(elems == elems[row,0])
         ind=ind[0]
         orderMax = np.max(orders[ind])

         # remove element if order exceeds threshold
         if (orderMax>threshold_order):
             elems[row,:]=-1

     # get rid of dud elements
     (elems, elem_properties)=remove_rows(elems, elem_properties)

     # reassign element numbers
     elemNum = np.zeros(len(elems))
     for i in range(0, len(elemNum)):
         elemNum[i]=i

    # put element properties back into geom format
     geom['elems']=np.column_stack([elemNum, elems])
     geom['length'] =elem_properties[0]
     geom['euclidean length'] = elem_properties[1]
     geom['radii'] = elem_properties[2]

     return (geom)


######
# Function: find branch angles, diameter ratio and length ratio of each branch
# Inputs: geom - contains elems, and various element properties (length, radius etc.)
#         orders - contains strahler order and generation of each element
#         elem_connect - contains upstream and downstream elements for each element
# Outputs: branch_angles - angle (radians) at each branching junction in the tree assigned to each element according to how it branches from its parent
#          diam_ratio - ratio of length/diameter of each branch, accounting for multi-segment branches
#          length_ratio - ratio of parent / child lengths, accounting for multi-segment branches
######

def find_branch_angles(geom, orders, elem_connect):

    # unpackage inputs
    nodes=geom['nodes']
    elems = geom['elems']
    elems = elems[:, 1:3]  # get rid of useless first column
    radii= geom['radii']
    lengths = geom['euclidean length']

    strahler=orders['strahler']
    generations = orders['generation']

    elem_up=elem_connect['elem_up']
    elem_down = elem_connect['elem_down']

    # new arrays
    num_elems = len(elems)
    branch_angles = -1. * np.ones(num_elems)
    diam_ratio = -1. * np.ones(num_elems)
    length_ratio = -1. * np.ones(num_elems)
    error = 0

    # find results for each element
    for ne in range(1, num_elems):

        # find parent
        neUp=elem_up[ne,1]

        if elem_up[ne,0]!=1:
            error=error+1 # expect each branch to have exactly one parent

        elif (generations[neUp] < generations[ne]) & (strahler[neUp] > strahler[ne]): # an actual branch causes both change in generation and order

            # parent node
            endNode=int(elems[neUp, 0])
            startNode=int(elems[neUp, 1])
            v_parent = nodes[endNode, :] - nodes[startNode,:]
            v_parent = v_parent / np.linalg.norm(v_parent)

            d_parent=2*radii[neUp]
            L_parent=lengths[neUp]

            # loop to find adjoining parents
            nextUp = elem_up[neUp, 1]
            continueLoop=1
            while (strahler[nextUp]==strahler[neUp])& continueLoop: # loop through to add joining branches
                L_parent=L_parent+lengths[nextUp]
                if elem_up[nextUp,0]>0:
                    nextUp=elem_up[nextUp,1]
                else:
                    continueLoop=0

            # daughter
            endNode = int(elems[ne, 1])
            startNode = int(elems[ne, 0])
            v_daughter = nodes[startNode, :] - nodes[endNode, :]
            v_daughter=v_daughter/np.linalg.norm(v_daughter)

            d_daughter=2*radii[ne]
            L_daughter=lengths[ne]

            # loop to find adjoining daughters
            nextDown = elem_down[ne, 1]
            continueLoop=1
            while (strahler[nextDown]==strahler[ne])& continueLoop: #loop through to add joining branches
                L_daughter=L_daughter+lengths[nextDown]
                if elem_down[nextDown,0]==1: #as can only be one daughter if unbranching
                    nextDown=elem_down[nextDown,1]
                else:
                    continueLoop=0

            # calculate results
            branch_angles[ne] = np.arccos(np.dot(v_parent, v_daughter))
            if d_parent!=0:
                diam_ratio[ne]=d_daughter/d_parent
            length_ratio[ne] = L_daughter / L_parent

    print('Number of elements for which no angle could be found (no unqiue parent) = ' + str(error))
    return (branch_angles, diam_ratio, length_ratio)


######
# Function: find statistics on branching tree and display as table
# Inputs: geom - contains various element properties (length, radius etc.) by element
#         orders - contains strahler order and generation of each element
# Outputs: table of information according to order and other information printed to screen
######

def summary_statistics(orders, geom):

    # unpack inputs
    strahler=orders['strahler']
    generation=orders['generation']

    length=geom['length']
    euclid_length = geom['euclidean length']
    radii = geom['radii']
    branch_angles = geom['branch_angles']
    diam_ratio = geom['diam_ratio']
    length_ratio = geom['length_ratio']

    # statisitcs by order
    num_orders = int(max(strahler))
    values_by_order=np.zeros([num_orders,9])
    extra_branches=0

    for n_ord in range(0, num_orders):

        elem_list = (strahler == n_ord+1)

        branch_list = np.extract(elem_list, branch_angles)
        branch_list = branch_list[(branch_list > -1)]
        Nb = len(branch_list)

        # case where there is one branch
        if Nb==0:
            Nb=1
            angle=np.nan
            extra_branches=extra_branches+1
        else:
            angle=np.mean(branch_list)
            
        # assign stats for each order
        values_by_order[n_ord,0]= n_ord + 1 # order
        values_by_order[n_ord, 1] = len(np.extract(elem_list, elem_list)) # number of segments
        values_by_order[n_ord, 2] = Nb # number of branches
        values_by_order[n_ord, 3] = np.sum(np.extract(elem_list, length))/ Nb # length
        values_by_order[n_ord, 4] = np.sum(np.extract(elem_list, euclid_length))/ Nb # euclidean length
        values_by_order[n_ord, 5] = 2*np.mean(np.extract(elem_list, radii))# diameter
        values_by_order[n_ord, 6] = values_by_order[n_ord, 3]/values_by_order[n_ord, 5] # length / diameter
        values_by_order[n_ord, 7] = np.mean(np.extract(elem_list, length)/np.extract(elem_list, euclid_length)) # tortuosity
        values_by_order[n_ord,8] = angle # branch angle

    # print table
    header = ['Order','# Segments','# Branches','Length(mm)','Euclidean Length(mm)', 'Diameter(mm)','Len/Diam', 'Tortuosity','Angle(degrees)']
    print('\n')
    print('Statistics By Order: ')
    print('..................')
    print(tabulate(values_by_order, headers=header))

    # statistics independent of order
    values_overall = np.zeros([1, 9])

    elem_list = (strahler > 0)
    branch_list = np.extract(elem_list, branch_angles)
    branch_list = branch_list[(branch_list > -1)]  # for actual distinct branches

    Nb = len(branch_list) + extra_branches


    values_overall[0, 0] = -1  # order
    values_overall[0, 1] = len(np.extract(elem_list, elem_list))  # number of segments
    values_overall[0, 2] = Nb # number of branches

    values_overall[0, 3] = np.sum(length) / Nb  # length
    values_overall[0, 4] = np.sum(euclid_length) / Nb  # euclidean length
    values_overall[0, 5] = 2 * np.mean(radii)  # diameter

    values_overall[0, 6] = values_overall[0, 4]/values_overall[0, 5]  # length / diameter
    values_overall[0, 7] = np.mean(np.extract(elem_list, length) / np.extract(elem_list, euclid_length))  # tortuosity
    values_overall[0, 8] = np.mean(branch_list) # branch angle

    header = ['     ', '          ', '          ', '          ', '                    ', '            ', '        ',
              '          ', '              ']
    print(tabulate(values_overall, headers=header))
    print('\n')

    # Other statistics
    print('Other statistics: ')
    print('..................')

    print('Num generations = ' + str(max(generation)))
    terminalGen = generation[(strahler == 1)]
    print('Average Terminal generation = ' + str(np.mean(terminalGen)))
    diam_ratio = diam_ratio[(diam_ratio > 0)]
    length_ratio = length_ratio[(length_ratio > -1)]
    print('D/Dparent = ' + str(np.mean(diam_ratio)))
    print('L/Lparent = ' + str(np.mean(length_ratio)))
    print('\n')

    return np.concatenate([values_by_order,values_overall])