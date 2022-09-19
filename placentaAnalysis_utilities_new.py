

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from mpl_toolkits.mplot3d import Axes3D
import skimage
from skimage import io
import pandas as pd
from PIL import Image
from statistics import mean
import nibabel as nib

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
    if Nc > 12:
        print('Warning, large number of elements at one node: ' + str(Nc))
        Nc = 12

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

###copied from old file just for this function
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

##################


def element_connectivity_1D(node_loc, elems, Nc) -> object:
    """
    @rtype: object
    """
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
        nnod2 = int(elems[ne][2])

        for noelem in range(1, elems_at_node[nnod2][0] + 1):
            ne2 = elems_at_node[nnod2][noelem]

            if ne2 != ne:
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
# Fucntion: Read in test answers for test script
# Inputs: reads in values from csv file ''test_tree_answers.csv'
# Outputs: puts all the answers as arrays into trueAnswers
######

def loadTestAnswers():

    #read csv
    with open('test_tree_answers.csv') as csvDataFile:
        cols=['nx','ny','nz','e0','e1','e2','e0_ordered','e1_ordered','e2_ordered','angle','diamR','lengthR','order','generation1','generation2','e0_ordered_2','e1_ordered_2','e2_ordered_2','branch','lenB','radiiB','angleB','diamRB','lengthRB']
        data_file = pd.read_csv(csvDataFile, usecols=cols)
        data_file.columns = cols

    #Convert answer files to arrays:
    generation1_true=data_file.generation1.values
    generation2_true=data_file.generation2.values
    diam_ratio_true=data_file.diamR.values
    length_ratio_true=data_file.lengthR.values
    order_true=data_file.order.values
    angles_true=data_file.angle.values
    branch_angles_true=data_file.angleB.values
    branch_Dratio_true = data_file.diamRB.values
    branch_Lratio_true = data_file.lengthRB.values
    branch_true=data_file.branch.values
    branch_len = data_file.lenB.values
    branch_radii = data_file.radiiB.values

    #package answers
    trueAnswers={}

    trueAnswers['order']=order_true[0:11]
    trueAnswers['generation1']=generation1_true[0:11]
    trueAnswers['generation2']=generation2_true[0:10]
    trueAnswers['angles']=angles_true[0:10]
    trueAnswers['diam_ratio']=diam_ratio_true[0:10]
    trueAnswers['length_ratio']=length_ratio_true[0:10]
    trueAnswers['branches'] = branch_true[0:10]
    trueAnswers['lengthB'] = branch_len[0:7]
    trueAnswers['radiiB'] = branch_radii[0:7]
    trueAnswers['diam_RatioB'] = branch_Dratio_true[0:7]
    trueAnswers['length_RatioB'] = branch_Lratio_true[0:7]
    trueAnswers['anglesB'] = branch_angles_true[0:7]

    cols2 = ['angle', 'diamR', 'lengthR','order', 'generation1', 'generation2', 'branch', 'lenB', 'radiiB', 'angleB', 'diamRB', 'lengthRB']
    data_file=data_file.drop(cols2, axis=1)
    data_file=data_file.values
    trueAnswers['nodes']=data_file[:,0:3]
    elems_unordered_true=data_file[:, 3:6]
    trueAnswers['elems_unordered']=elems_unordered_true[0:11,:]
    elems_ordered_true=data_file[:, 6:9]
    trueAnswers['elems_ordered']=elems_ordered_true[0:11,:]
    elems_pruned_true=data_file[:, 9:12]
    trueAnswers['elems_pruned']=elems_pruned_true[0:10,:]

    return trueAnswers

######
# Function: Loads in a stack of images, located in path, and with naming convention name (goes slice at a time to avoid memory errors)
#     Inputs: numImages - integer, number of images in the stack
#             name - string for name of images. Note images must be numbered from 0
#     Outputs: Image, a 3D BOOLEAN array containing image
######

def load_image_bool(name, numImages):

    # read in first image + get dimensions to initialize array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = io.imread(name.format(0))
        skimage.img_as_bool(im)
    Image=np.zeros([im.shape[0], im.shape[1], numImages], dtype=bool)
    Image[:, :, 0] = im

    # load all slices
    for i in range(1, numImages):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = io.imread(name.format(i))
            skimage.img_as_bool(im)
        Image[:,:,i]=im

    print('Image ' + name + ' loaded. Shape: ' + str(Image.shape))
    return Image

def load_image_nifti(name, numImages):

    nifti_file = nib.load(name)
    nifti_stack = nifti_file.get_fdata()

    if len(nifti_stack.shape) == 4:
        new_stack = np.squeeze(nifti_stack,axis=3)
    Image = np.array(new_stack)

    Image = np.swapaxes(Image,0,1)


    # # read in first image + get dimensions to initialize array
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     im = io.imread(name.format(0))
    #     skimage.img_as_bool(im)
    # Image=np.zeros([im.shape[0], im.shape[1], numImages], dtype=bool)
    # Image[:, :, 0] = im
    #
    # # load all slices
    # for i in range(1, numImages):
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         im = io.imread(name.format(i))
    #         skimage.img_as_bool(im)
    #     Image[:,:,i]=im

    print('Image ' + name + ' loaded. Shape: ' + str(Image.shape))
    return Image

######
# Function: Finds Strahler ratio of variable
#     Inputs: Orders- an array containing the orders of the vascular tree
#             Factor - an array containing a value for each order of the tree e.g. number of branches at each order
#     Outputs: Strahler ratio e.g. Rb, Rd for that factor and the R^2 value of the linear fit used to produce it
######

def find_strahler_ratio(Orders, Factor):

    x = Orders
    yData = np.log(Factor)
    plt.plot(x, yData, 'k--', linewidth=1.5, label='Data')

    # fit line to data
    xFit = np.unique(Orders)
    yFit = np.poly1d(np.polyfit(x, yData, 1))(np.unique(x))
    plt.plot(np.unique(x), yFit, label='linear fit')

    # Scaling Coefficient is gradient of the fit
    grad = (yFit[len(yFit) - 1] - yFit[0]) / (xFit[len(xFit) - 1] - xFit[0])
    grad=np.abs(grad)
    grad=np.exp(grad)

    # R^2 value
    yMean = [mean(yData) for y in yData]
    r2 = 1 - (sum((yFit - yData) * (yFit - yData)) / sum((yMean - yData) * (yMean - yData)))

    heading = ('Strahler Ratio = ' + str(grad))
    plt.title(heading)
    plt.legend()
    plt.show()

    return grad, r2
