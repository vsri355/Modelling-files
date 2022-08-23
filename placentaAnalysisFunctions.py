import numpy as np
from tabulate import tabulate
from placentaAnalysis_utilities import *
import scipy
import scipy.spatial
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from ellipsoid import*
from statistics import mean

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
    data_file = data_file[data_file.SkeletonID == 1]

    # get skeleton properties as arrays
    euclid_length = data_file.Euclideandistance.values
    length = data_file.Branchlength.values
    radii = data_file.averageintensityinner3rd.values

    # get elem and node data
    data_file = data_file.drop(['SkeletonID', 'Branchlength', 'averageintensityinner3rd', 'Euclideandistance'], axis=1)
    data_file = data_file.values
    (elems, nodes) = sort_elements(data_file[:, 0:3], data_file[:, 3:6])

    # get rid of dud elements
    (elems, [length, euclid_length, radii]) = remove_rows(elems, [length, euclid_length, radii])

    return {'nodes': nodes, 'elems': elems, 'radii': radii, 'length': length, 'euclidean length': euclid_length}


######
# Function: takes a list of node pairs (v1, v2) and creates a list of nodes and elements
# Inputs: v1 - N x 3 array of start node coordinates
#         v2 - N x 3 array of end node coordinates
# Outputs: nodes - an M x 3 array giving cartesian coordinates (x,y,z) for the node locations in the tree
#          elems - an N x 3 array, the first colum in the element number, the second two columns are the index of the start and end node
#          elements that start and end at the same node are given a value of -1
######

def sort_elements(v1, v2):
    Nelem = len(v1)
    elems = np.zeros([Nelem, 3])
    nodes = np.zeros([Nelem * 2, 3])  # max number of nodes possible

    iN = 0  # node index

    # go through first node list
    for iE in range(0, Nelem):

        v = v1[iE, :]
        index = is_member(v, nodes[0:iN][:])  # see if the node is in the nodes list

        if index == -1:  # if not, create a new node
            nodes[iN, :] = v
            index = iN
            iN = iN + 1

        # else just use index of existing node
        elems[iE, 1] = int(index)
        elems[iE, 0] = int(iE)  # first column of elements is just the element number

    # go through second node list
    for iE in range(0, Nelem):

        v = v2[iE, :]
        index = is_member(v, nodes[0:iN, :])

        if index == -1:
            nodes[iN, :] = v
            index = iN
            iN = iN + 1

        elems[iE, 2] = int(index)

        if elems[iE][1] == elems[iE][2]:
            elems[iE, 0:2] = int(-1)

    nodes = nodes[0:iN:1][:]  # truncate based on how many nodes were actually assigned

    return (elems, nodes)


######
# Function: rearranges elems (and corresponding properties) according to their strahler order, to be compatible with placentagen functions
# Inputs: geom - contains elems, nodes and other properties of the skeleton
#         inlet_loc - the coordinates of the parent node for the entire tree (if known)
#         find_inlet_loc - a boolean variable specifying whether to use inlet location provided (0) or to find the inlet location automatically (1)
# Outputs: geom - contains elems and properties, reordered according to strahler order so that no element can be higher in the element list than a higher order branch
######

def arrange_by_strahler_order(geom, find_inlet_loc, inlet_loc):
    # set up arrays
    nodes = geom['nodes']
    elem_properties = np.column_stack([geom['radii'], geom['length'], geom['euclidean length'], geom['elems']])
    elems = np.copy(geom['elems'])  # as elems is altered in this function
    elems = elems[:, 1:3]  # get rid of first column which means nothing
    radii = geom['radii']

    Ne = len(elems)
    Nn = len(nodes)
    elem_properties_new = np.zeros([Ne, 6])

    # find parent node
    (elems, elem_properties) = find_parent_node(find_inlet_loc, inlet_loc, nodes, radii, elems, elem_properties)

    # loop through by strahler order
    counter_new = 0
    counter = 1
    while (counter < Ne):

        # find elements which are terminal
        terminal_elems = np.zeros([Ne, 1])

        # go through each node
        for i in range(0, Nn + 1):

            # find number of occurrences of the node
            places = np.where(elems == i)
            ind1 = places[0]
            ind2 = places[1]

            if (len(ind1) == 1) and ((ind1[0]) != 0):  # if occurs once, then element is terminal (avoids root element)

                ind1 = ind1[0]
                ind2 = ind2[0]

                # swap to ensure element points right way
                if ind2 == 0:
                    elems[ind1, :] = row_swap_1d(np.squeeze(elems[ind1, :]), 1, 0)
                    elem_properties[ind1, 4:6] = row_swap_1d(np.squeeze(elem_properties[ind1, 4:6]), 1, 0)

                # assign element under the new element ordering scheme
                elem_properties_new[counter_new, :] = elem_properties[ind1, :]
                counter_new = counter_new + 1

                terminal_elems[ind1] = 1

                # join up element with upstream elements
                nodeNumNew = elems[ind1, 0]  # this is node number at other end of element
                nodeNum = i
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
                    terminal_elems[ind1[k]] = 1  # label terminal_elems as joining elements

                    # switch the way element points
                    if (ind2[k] == 0):
                        elems[ind1[k], :] = row_swap_1d(np.squeeze(elems[ind1[k], :]), 1, 0)
                        elem_properties[ind1[k], 4:6] = row_swap_1d(np.squeeze(elem_properties[ind1[k], 4:6]), 1, 0)

                    nodeNum = nodeNumNew
                    nodeNumNew = elems[ind1[k], 0]

                    # assign new order
                    elem_properties_new[counter_new, :] = elem_properties[ind1[k], :]
                    counter_new = counter_new + 1

                    # update loop criteria
                    places = np.where(elems == nodeNumNew)
                    ind1 = places[0]
                    ind2 = places[1]
                    counter2 = counter2 + 1

        # update elems to 'get rid of' terminal elements from the list
        terminal_elems[0] = 0  # the root node can never be terminal
        terminal_elems_pair = np.column_stack([terminal_elems, terminal_elems])
        elems[terminal_elems_pair == 1] = -1

        # loop exit criteria
        places = np.where(terminal_elems == 1)
        places = places[1]
        if len(places) == 0:
            counter = Ne + 1
        counter = counter + 1

    # assign root element in new order systems
    elem_properties_new[Ne - 1, :] = elem_properties[0, :]

    # reduce size due to elements removed
    elem_properties_new = elem_properties_new[0:Ne, :]
    # reverse order
    elem_properties_new = np.flip(elem_properties_new, 0)

    elems = geom['elems']
    elems = elems[0:Ne, :]
    elems[:, 1:3] = elem_properties_new[:, 4:6]
    radii = elem_properties_new[:, 0]
    lengths = elem_properties_new[:, 1]
    euclid_lengths = elem_properties_new[:, 2]

    return {'elems': elems, 'radii': radii, 'length': lengths, 'euclidean length': euclid_lengths, 'nodes': nodes}


######
# Function: find parent node in array either from given coordinates or by finding the fattest terminal branch
# Inputs: nodes - an M x 3 list of node coordinates
#        radii - N x 1 array with radius of each element
#         elems - an Nx2(!!!) array with node indices for start and end of node
#         elem_properties - an N x K array, with each row containing various element properties (radii etc.)
#         inlet_loc - the coordinates of the parent node for the entire tree (if known)
#         find_inlet_loc - a boolean variable specifying whether to use inlet location provided (0) or to find the inlet location automatically (1)
# Outputs: elems and elem_properties updates so that inlet element is the first element in the list
######
def find_parent_node(find_inlet_loc, inlet_loc, nodes, radii, elems, elem_properties):
    # will define inlet as terminal element of largest radius
    if find_inlet_loc == 1:
        maxRad = -1
        # go through each node
        for i in range(0, len(nodes) + 1):

            # find number of occurrences of the node
            places = np.where(elems == i)
            ind1 = places[0]
            ind2 = places[1]

            if (len(ind1) == 1):  # if occurs once, then element is terminal (avoids root element)

                ind1 = ind1[0]
                ind2 = ind2[0]
                radius = radii[ind1]

                if radius > maxRad:
                    maxRad = radius
                    maxRadInd = i

        inlet_loc = np.squeeze(nodes[maxRadInd, :])
        Nn_root = maxRadInd
    # find root node and element from coordinates provided
    else:
        Nn_root = is_member(inlet_loc, nodes)
        if (Nn_root == -1):
            print("Warning, root node not located")

    print('Inlet Coordinates:' + str(inlet_loc))

    # find root element
    Ne_place = np.where(elems == Nn_root)
    Ne_root = Ne_place[0]  # only need first index
    if len(Ne_root) > 1:
        print("Warning, root node is associated with multiple elements")
    if len(Ne_root) == 0:
        print("Warning, no root element located")
    Ne_root = Ne_root[0]

    # make root element the first element
    elems = row_swap_2d(elems, 0, Ne_root)
    elem_properties = row_swap_2d(elem_properties, 0, Ne_root)

    # get element pointing right way
    if (np.squeeze(Ne_place[1]) != 0):
        elems[0, :] = row_swap_1d(np.squeeze(elems[0, :]), 1, 0)
        elem_properties[0, 4:6] = row_swap_1d(np.squeeze(elem_properties[0, 4:6]), 1, 0)

    return (elems, elem_properties)


######
# Function: Two types of pruning
#            - removes terminal branches that connect directly to high order branches
#            - removes low order sections of tree that connect directly to the umbilical artery
# Inputs: geom - contains elems, and various element properties (length, radius etc.)
#         orders - contains strahler order of each element orders
#         threshold_order - the maximum order which a terminal element can be directly connected to
#         umbilical_threshold - the lowest order branch that can stem from the umbilical artery
#         Nc - the maximum number of branches at a node
# Outputs: elems and their corresponding properties are truncated to remove the rows containing pruned elements
######
def prune_by_order(geom, elem_connect, orders, threshold_order, umbilical_threshold, Nc):
    # define arrays
    elem_properties = [geom['length'], geom['euclidean length'], geom['radii'], orders]
    elem_down = elem_connect['elem_down']
    elems = geom['elems']
    elems = elems[:, 1:3]
    Ne = len(elems)

    change = 0

    # go through list of terminal elements to remove terminal elements adjoining high order brnaches
    terminalList = np.where(orders == 1)
    terminalList = terminalList[0]

    for i in range(0, len(terminalList)):
        row = terminalList[i]

        # find parents at the non terminal end of the element, and their order
        ind = np.where(elems == elems[row, 0])
        ind = ind[0]
        orderMax = np.max(orders[ind])

        # remove element if order exceeds threshold
        if (orderMax > threshold_order):
            elems[row, :] = -1
            change = 1

    # remove daughters of root element that have order lower than umbilical_threshold
    for i in range(1, elem_down[0, 0] + 1):

        daughter = elem_down[0, i]
        order_daughter = orders[daughter]

        if order_daughter < umbilical_threshold:
            elems[daughter, :] = -1
            change = 1

    # loop to remove any sections that have been isolated from tree
    numIt = 0
    while (change == 1) & (numIt < Ne):

        change = 0
        numIt = numIt + 1

        # get rid of dud elements
        (elems, elem_properties) = remove_rows(elems, elem_properties)
        orders = elem_properties[3]
        elemNum = np.zeros(len(elems))
        for i in range(0, len(elemNum)):
            elemNum[i] = i

        # update connectivity
        elem_connect = element_connectivity_1D(geom['nodes'], np.column_stack([elemNum, elems]), Nc)
        elem_up = elem_connect['elem_up']

        # check for branches with no upstream element (therefore a disconnected tree)
        for i in range(1, len(elem_up)):
            if elem_up[i, 0] == 0:
                if orders[i] != np.max(orders):  # don't get rid of parent node
                    elems[i, :] = -1
                    change = 1
                else:
                    print('keep if statement')

    # put element properties back into geom format
    geom['elems'] = np.column_stack([elemNum, elems])
    geom['length'] = elem_properties[0]
    geom['euclidean length'] = elem_properties[1]
    geom['radii'] = elem_properties[2]

    return (geom, elem_connect)

######
# Function: finds properties of according to each Branch of the tree, where a branch is a set of elements with the
#          same Strahler order
# Inputs: geom - contains elems, and various element properties (length, radius etc.)
#         order - contains strahler order and generation of each element
#         elem_up - contains index of upstream elements for each element
# Outputs: branchGeom: contains the properties arrange in arrays according to each branch:
#           radius / length / euclidean length / strahler order: all M x 1 arrays where M is number of branches
#          branches: an N x 1 array where N is the number of elements, contains branch number of each element
######

def arrange_by_branches(geom, elem_up, order):

    # find branches
    Ne = len(order)
    branches = np.zeros(Ne)
    branchNum = 1

    for i in range(0, Ne):
        if order[i] != order[elem_up[i, 1]]:  # belongs to new branch
            branchNum = branchNum + 1
        branches[i] = branchNum

    Nb = int(max(branches))

    # sort results into branch groups
    lengths = geom['length']
    radii = geom['radii']
    nodes= geom['nodes']
    elems = geom['elems']

    branchRad = np.zeros(Nb)
    branchLen = np.zeros(Nb)
    branchEucLen = np.zeros(Nb)
    branchOrder = -1. * np.ones(Nb)

    for i in range(0, Nb):
        branchElements = np.where(branches == i+1) #find elements belonging to branch number
        branchElements = branchElements[0]

        for j in range(0, len(branchElements)): #go through all elements in branch
            ne = branchElements[j]

            branchOrder[i] = order[ne]
            branchLen[i] = branchLen[i] + lengths[ne]
            branchRad[i] = branchRad[i] + radii[ne]

        branchRad[i] = branchRad[i] / len(branchElements) # to get average radius

        startNode=nodes[int(elems[branchElements[0],1]),:]
        endNode=nodes[int(elems[branchElements[len(branchElements)-1],2]),:]

        branchEucLen[i]=np.sqrt(np.sum(np.square(startNode-endNode)))

    return {'radii': branchRad, 'length': branchLen, 'euclidean length': branchEucLen, 'order': branchOrder,
            'branches': branches}


######
# Function: find branch angles + L/LParent & D/Dparent
#          scale all results into mm and degrees
# Inputs: geom - contains elems, and various element properties (length, radius etc.)
#         orders - contains strahler order and generation of each element
#         elem_connect - contains upstream and downstream elements for each element
#         branchGeom - contains branch properties (length, radius, etc.)
#         voxelSize - for conversion to mm (must be isotropic)
#         conversionFactor - to scale radii correction, printed in log of ImageJ during MySkeletonizationProcess
# Outputs: geom and branchGeom are altered so all there arrays are in correct units (except nodes, and radii_unscaled, which remain in voxels) ##################
#          seg_angles - angle (radians) at each element junction in the tree assigned to each element according to how it branches from its parent
#          diam_ratio - ratio of length/diameter of each branch, accounting for multi-segment branches
#          length_ratio - ratio of parent / child lengths, accounting for multi-segment branches
#          diam_ratio / length_ratio / branch_angles are the same but for whole branches
######

def find_branch_angles(geom, orders, elem_connect, branchGeom, voxelSize, conversionFactor):

    # unpackage inputs
    nodes = geom['nodes']
    elems = geom['elems']
    elems = elems[:, 1:3]  # get rid of useless first column
    radii = geom['radii']
    lengths = geom['length']

    branches = branchGeom['branches']
    branchRad = branchGeom['radii']
    branchLen = branchGeom['length']

    strahler = orders['strahler']
    generations = orders['generation']

    elem_up = elem_connect['elem_up']

    # new arrays
    num_elems = len(elems)
    num_branches = len(branchRad)

    branch_angles = -1. * np.ones(num_branches) # results by branch (Strahler)
    diam_ratio_branch = -1. * np.ones(num_branches)
    length_ratio_branch = -1. * np.ones(num_branches)

    diam_ratio = -1. * np.ones(num_elems)  # results by generation
    length_ratio = -1. * np.ones(num_elems)
    seg_angles = -1. * np.ones(num_elems)

    # find results for each element (ignoring parent element)
    for ne in range(1, num_elems):

        neUp = elem_up[ne, 1] # find parent

        if (generations[neUp] < generations[ne]): # there is branching but not necessarily a new strahler branch

            # parent node
            endNode = int(elems[neUp, 0])
            startNode = int(elems[neUp, 1])
            v_parent = nodes[endNode, :] - nodes[startNode, :]
            v_parent = v_parent / np.linalg.norm(v_parent)

            d_parent = 2 * radii[neUp]
            L_parent = lengths[neUp]

            # daughter
            endNode = int(elems[ne, 1])
            startNode = int(elems[ne, 0])
            v_daughter = nodes[startNode, :] - nodes[endNode, :]
            v_daughter = v_daughter / np.linalg.norm(v_daughter)

            d_daughter = 2 * radii[ne]
            L_daughter = lengths[ne]

            # calculate angle
            dotProd = np.dot(v_parent, v_daughter)
            if abs(dotProd <= 1):
                angle=np.arccos(dotProd)
                seg_angles[ne] = angle
            else:
                angle=-1
                print('Angle Error, element: ' + str(ne))

            if d_parent != 0:
                diam_ratio[ne] = d_daughter/ d_parent
            if L_parent != 0:
                length_ratio[ne] = L_daughter / L_parent

            if (strahler[neUp] > strahler[ne]): #then this also is a new strahler branch

                # assign results
                branchNum = int(branches[ne])-1
                parentBranch = int(branches[neUp])-1

                branch_angles[branchNum] = angle

                if branchRad[parentBranch] != 0:
                    diam_ratio_branch[branchNum] = branchRad[branchNum] / branchRad[parentBranch]
                if branchLen[parentBranch] != 0:
                    length_ratio_branch[branchNum] = branchLen[branchNum] / branchLen[parentBranch]

    # scale results into mm and degrees & package them up
    geom['radii'] = geom['radii'] / conversionFactor * voxelSize
    geom['length'] = geom['length'] * voxelSize
    geom['nodes'] = geom['nodes'] * voxelSize
    geom['euclidean length'] = geom['euclidean length'] * voxelSize
    geom['branch angles'] = seg_angles * 180 / np.pi
    geom['diam_ratio'] = diam_ratio
    geom['length_ratio'] = length_ratio

    branchGeom['radii']= branchGeom['radii']/ conversionFactor
    branchGeom['radii'] = branchGeom['radii'] * voxelSize
    branchGeom['branch_angles'] = branch_angles * 180 / np.pi
    branchGeom['length'] = branchGeom['length'] * voxelSize
    branchGeom['euclidean length'] = branchGeom['euclidean length'] * voxelSize

    branchGeom['length ratio'] = length_ratio_branch
    branchGeom['diam ratio'] = diam_ratio_branch

    return (geom, branchGeom)


#######################
# Function: Find the Major/Minor ratios of length, diameter and branch angle
# Inputs:  geom - contains elements, and their radii, angles and lengths
#          elem_down - contains the index of the downstream elements at each element
# Outputs: grad - the diameter scaling coefficient
#######################

def major_minor(geom, elem_down):

    # extract data
    radii=geom['radii']
    angles=geom['branch angles']
    length=geom['length']

    # create arrays
    Ne=len(elem_down)

    Minor_angle=-1*np.ones(Ne)
    Major_angle = -1*np.ones(Ne)

    D_Major_Minor = -1 * np.ones(Ne)
    D_min_parent = -1 * np.ones(Ne)
    D_maj_parent = -1 * np.ones(Ne)

    L_Major_Minor = -1 * np.ones(Ne)
    L_min_parent = -1 * np.ones(Ne)
    L_maj_parent = -1 * np.ones(Ne)

    for i in range(0, Ne):
        numDown=elem_down[i, 0]

        if numDown>1: # then this element has multiple children, find minor / major child

            d_min=100000
            d_max=0
            for j in range(1, numDown+1): #look throigh children and find widest & thinnest one
                child=np.int(elem_down[i, j])
                d_child=radii[child]

                if d_child>d_max:
                    d_max=d_child
                    daughter_max=child
                if d_child<d_min:
                    d_min = d_child
                    daughter_min = child

            if daughter_max!=daughter_min: # ensure two distinct daughters

                Minor_angle[i]=angles[daughter_min]
                Major_angle[i]=angles[daughter_max]

                if radii[daughter_min]!=0: # avoid divide by zero errors
                    D_Major_Minor[i]=radii[daughter_max]/radii[daughter_min]
                if radii[i] != 0:
                    D_min_parent[i]=radii[daughter_min]/radii[i]
                    D_maj_parent[i]=radii[daughter_max]/radii[i]

                if length[daughter_min] != 0:
                    L_Major_Minor[i] = length[daughter_max] / length[daughter_min]
                if length[i] != 0:
                    L_min_parent[i] = length[daughter_min] / length[i]
                    L_maj_parent[i] = length[daughter_max] / length[i]
    return {'Minor_angle': Minor_angle, 'Major_angle': Major_angle, 'D_maj_min': D_Major_Minor, 'D_min_P': D_min_parent,'D_maj_P': D_maj_parent, 'L_maj_min': L_Major_Minor, 'L_min_P': L_min_parent,'L_maj_P': L_maj_parent}


#######################
# Function: Find & print vascular depth, span & volume
# Inputs:  nodes - M x 3 array with node coordinates
#          elems - N x 3 array containing elements
#          orders - N x 1 array with order of each element
#          vascVol - vascularVolume, in mm^3 (number of voxels in the volume image)
# Outputs: depth- Vascular depth, mm
#          span, vascular span, mm
#######################

def overall_shape_parameters(nodes, elems,orders, vascVol):

    # find umbilical insertion node
    inds=np.where(orders==np.max(orders))
    inds=inds[0]
    inds=inds[len(inds)-1] #last umbilical elemnt should be further down
    umbNode=np.int(elems[inds,2]) #take second node as elems point downstream
    umbEnd=nodes[umbNode,:]

    # extract termimal nodes
    inds = np.where(orders == 1)
    inds = inds[0]
    endNodes=elems[inds,2] #take second node as elems point downstream
    endNodes=np.squeeze(endNodes.astype(int))
    endPoints=nodes[endNodes,:]

    # Vascular Span
    dists=scipy.spatial.distance.pdist(endPoints, 'euclidean') # pairwise distance between points
    span=(np.max(dists))

    # Get placenta volume
    ET = EllipsoidTool()
    (center, radii, rotation) = ET.getMinVolEllipse(endPoints, .01)
    ellipseVol = ET.getEllipsoidVolume(radii)
    ET.plotEllipsoid(center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2)

    print('\nOverall Placenta Shape')
    print('-----------------------')
    print("Vascular Span = " + str(span) + ' mm')
    print('Vascular Volume = ' + str(vascVol) + ' mm^3' )
    print('Placenta Volume = ' + str(ellipseVol)+ ' mm^3' )
    print("Vascular Density = " + str(vascVol/ellipseVol))
    print('\n')
    return span
