import numpy as np
from tabulate import tabulate
from placentaAnalysis_utilities import *
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from statistics import mean

####
# Function: find statistics on branching tree and display as table, sorting my generations in the tree
# Inputs: geom - contains various element properties (length, radius etc.) by element
#         orders - contains strahler order and generation of each element
# Outputs: table of information according to generation prints to screen
######

def generation_summary_statistics(geom, orders, major_minor_results):

    # unpack inputs
    generation = orders['generation']

    diam = 2 * geom['radii']
    length = geom['length']
    euclid_length = geom['euclidean length']
    angles = geom['branch angles']

    diam_ratio = geom['diam_ratio']
    length_ratio = geom['length_ratio']

    Minor_angle = major_minor_results['Minor_angle']
    Major_angle = major_minor_results['Major_angle']

    D_Major_Minor = major_minor_results['D_maj_min']
    D_min_parent = major_minor_results['D_min_P']
    D_maj_parent = major_minor_results['D_maj_P']

    L_Major_Minor = major_minor_results['L_maj_min']
    L_min_parent = major_minor_results['L_min_P']
    L_maj_parent = major_minor_results['L_maj_P']

    # statisitcs by generation
    num_gens= int(max(generation))
    values_by_gen = np.zeros([num_gens, 34])

    for n_gen in range(0, num_gens):

        element_list = (generation == n_gen + 1)

        diam_list = np.extract(element_list, diam)
        len_list = np.extract(element_list, length)

        # account for zero diameters
        diam_bool = diam_list > 0
        len_bool = len_list > 0
        list = np.logical_and(diam_bool, len_bool)
        diam_list = diam_list[list]
        len_list = len_list[list]

        # assign stats for each order
        values_by_gen[n_gen, 0] = n_gen + 1  # order
        values_by_gen[n_gen, 1] = len(np.extract(element_list, element_list))  # number of branches

        values_by_gen[n_gen, 2] = np.mean(np.extract(element_list, length))  # length
        values_by_gen[n_gen, 3] = np.std(np.extract(element_list, length))  # length std

        values_by_gen[n_gen, 4] = np.mean(diam_list)  # diameter
        values_by_gen[n_gen, 5] = np.std(diam_list)  # diameter std

        values_by_gen[n_gen, 6] = np.mean(np.extract(element_list, euclid_length))  # euclidean length
        values_by_gen[n_gen, 7] = np.std(np.extract(element_list, euclid_length))  # euclidean length std

        values_by_gen[n_gen, 8] = np.mean(len_list / diam_list)  # length / diameter
        values_by_gen[n_gen, 9] = np.std(len_list / diam_list)  # length / diameter std

        values_by_gen[n_gen, 10] = np.mean(
            np.extract(element_list, length) / np.extract(element_list, euclid_length))  # tortuosity
        values_by_gen[n_gen, 11] = np.std(
            np.extract(element_list, length) / np.extract(element_list, euclid_length))  # tortuosity


        if n_gen > 0:


            angle_list = np.extract(element_list, angles)
            angle_list = angle_list[angle_list > 0]
            if len(angle_list)>0:
                values_by_gen[n_gen, 12] = np.mean(angle_list)  # angles
                values_by_gen[n_gen, 13] = np.std(angle_list)  # angles std

            Minor_angle_list = np.extract(element_list, Minor_angle)
            Minor_angle_list = Minor_angle_list[Minor_angle_list > 0]
            Major_angle_list = np.extract(element_list, Major_angle)
            Major_angle_list = Major_angle_list[Major_angle_list > 0]
            if len(Minor_angle_list) > 0:
                values_by_gen[n_gen, 14] = np.mean(Minor_angle_list)  # minor angles
                values_by_gen[n_gen, 15] = np.std(Minor_angle_list)
                values_by_gen[n_gen, 16] = np.mean(Major_angle_list)  # major angles
                values_by_gen[n_gen, 17] = np.std(Major_angle_list)

            lengthRatio = np.extract(element_list, length_ratio)
            lengthRatio = lengthRatio[lengthRatio > 0]

            L_min_parent_list = np.extract(element_list, L_min_parent)
            L_min_parent_list = L_min_parent_list[L_min_parent_list > 0]
            L_maj_parent_list = np.extract(element_list, L_maj_parent)
            L_maj_parent_list = L_maj_parent_list[L_maj_parent_list > 0]
            L_Major_Minor_list = np.extract(element_list, L_Major_Minor)
            L_Major_Minor_list = L_Major_Minor_list[L_Major_Minor_list > 0]
            if len(L_min_parent_list) > 0:
                values_by_gen[n_gen, 18] = np.mean(lengthRatio)  # len ratio
                values_by_gen[n_gen, 19] = np.std(lengthRatio)  # len ratio
                values_by_gen[n_gen, 20] = np.mean(L_min_parent_list)
                values_by_gen[n_gen, 21] = np.std(L_min_parent_list)
                values_by_gen[n_gen, 22] = np.mean(L_maj_parent_list)
                values_by_gen[n_gen, 23] = np.std(L_maj_parent_list)
                values_by_gen[n_gen, 24] = np.mean(L_Major_Minor_list)
                values_by_gen[n_gen, 25] = np.std(L_Major_Minor_list)

            diamRatio = np.extract(element_list, diam_ratio)
            diamRatio = diamRatio[diamRatio > 0]
            D_min_parent_list = np.extract(element_list, D_min_parent)
            D_min_parent_list = D_min_parent_list[D_min_parent_list > 0]
            D_maj_parent_list = np.extract(element_list, D_maj_parent)
            D_maj_parent_list = D_maj_parent_list[D_maj_parent_list > 0]
            D_Major_Minor_list = np.extract(element_list, D_Major_Minor)
            D_Major_Minor_list = D_Major_Minor_list[D_Major_Minor_list > 0]
            if len(D_min_parent_list) > 0:
                values_by_gen[n_gen, 26] = np.mean(diamRatio)  # diam ratio
                values_by_gen[n_gen, 27] = np.std(diamRatio)  # diam std
                values_by_gen[n_gen, 28] = np.mean(D_min_parent_list)
                values_by_gen[n_gen, 29] = np.std(D_min_parent_list)
                values_by_gen[n_gen, 30] = np.mean(D_maj_parent_list)
                values_by_gen[n_gen, 31] = np.std(D_maj_parent_list)
                values_by_gen[n_gen, 32] = np.mean(D_Major_Minor_list)
                values_by_gen[n_gen, 33] = np.std(D_Major_Minor_list)

    # print table
    header = ['Gen', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'Euclidean Length(mm)', 'std',
              'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std','Minor Angle','std','Major Angle','std', 'LLparent', 'std', 'LminLparent', 'std', 'LmajLparent', 'std', 'LminLmaj', 'std', 'DDparent', 'std','DminDparent', 'std','DmajDparent', 'std','DminDmaj', 'std']
    print('\n')
    print('Statistics By Generation: ')
    print('..................')
    print(tabulate(values_by_gen, headers=header))

    # statistics independent of order
    values_overall = np.zeros([1, 34])

    element_list = (generation > 0)
    diam_list = np.extract(element_list, diam)

    len_list = np.extract(element_list, length)
    len_list = len_list[diam_list > 0]
    diam_list = diam_list[diam_list > 0]

    angle_list = np.extract(element_list, angles)
    angle_list = angle_list[angle_list > 0]

    Minor_angle_list = np.extract(element_list, Minor_angle)
    Minor_angle_list = Minor_angle_list[Minor_angle_list > 0]
    Major_angle_list = np.extract(element_list, Major_angle)
    Major_angle_list = Major_angle_list[Major_angle_list > 0]
    L_min_parent_list = np.extract(element_list, L_min_parent)
    L_min_parent_list = L_min_parent_list[L_min_parent_list > 0]
    L_maj_parent_list = np.extract(element_list, L_maj_parent)
    L_maj_parent_list = L_maj_parent_list[L_maj_parent_list > 0]
    L_Major_Minor_list = np.extract(element_list, L_Major_Minor)
    L_Major_Minor_list = L_Major_Minor_list[L_Major_Minor_list > 0]
    D_min_parent_list = np.extract(element_list, D_min_parent)
    D_min_parent_list = D_min_parent_list[D_min_parent_list > 0]
    D_maj_parent_list = np.extract(element_list, D_maj_parent)
    D_maj_parent_list = D_maj_parent_list[D_maj_parent_list > 0]
    D_Major_Minor_list = np.extract(element_list, D_Major_Minor)
    D_Major_Minor_list = D_Major_Minor_list[D_Major_Minor_list > 0]

    # assign stats for each order
    values_overall[0, 0] = -1
    values_overall[0, 1] = len(np.extract(element_list, element_list))  # number of branches

    values_overall[0, 2] = np.mean(len_list)  # length
    values_overall[0, 3] = np.std(len_list)  # length std

    values_overall[0, 4] = np.mean(diam_list)  # diameter
    values_overall[0, 5] = np.std(diam_list)  # diameter std

    values_overall[0, 6] = np.mean(np.extract(element_list, euclid_length))  # euclidean length
    values_overall[0, 7] = np.std(np.extract(element_list, euclid_length))  # euclidean length std

    values_overall[0, 8] = np.mean(len_list / diam_list)  # length / diameter
    values_overall[0, 9] = np.std(len_list / diam_list)  # length / diameter std

    values_overall[0, 10] = np.mean(
        np.extract(element_list, length) / np.extract(element_list, euclid_length))  # tortuosity
    values_overall[0, 11] = np.std(
        np.extract(element_list, length) / np.extract(element_list, euclid_length))  # tortuosity

    values_overall[0, 12] = np.mean(angle_list)  # angles
    values_overall[0, 13] = np.std(angle_list)  # angles std
    values_overall[0, 14] = np.mean(Minor_angle_list)  # minor angles
    values_overall[0, 15] = np.std(Minor_angle_list)
    values_overall[0, 16] = np.mean(Major_angle_list)  # major angles
    values_overall[0, 17] = np.std(Major_angle_list)

    lengthRatio = np.extract(element_list, length_ratio)
    lengthRatio = lengthRatio[lengthRatio > 0]
    values_overall[0, 18] = np.mean(lengthRatio)  # len ratio
    values_overall[0, 19] = np.std(lengthRatio)  # len ratio
    values_overall[0, 20] = np.mean(L_min_parent_list)
    values_overall[0, 21] = np.std(L_min_parent_list)
    values_overall[0, 22] = np.mean(L_maj_parent_list)
    values_overall[0, 23] = np.std(L_maj_parent_list)
    values_overall[0, 24] = np.mean(L_Major_Minor_list)
    values_overall[0, 25] = np.std(L_Major_Minor_list)

    diamRatio = np.extract(element_list, diam_ratio)
    diamRatio = diamRatio[diamRatio > 0]
    values_overall[0, 26] = np.mean(diamRatio)  # diam ratio
    values_overall[0, 27] = np.std(diamRatio)  # diam std
    values_overall[0, 28] = np.mean(D_min_parent_list)
    values_overall[0, 29] = np.std(D_min_parent_list)
    values_overall[0, 30] = np.mean(D_maj_parent_list)
    values_overall[0, 31] = np.std(D_maj_parent_list)
    values_overall[0, 32] = np.mean(D_Major_Minor_list)
    values_overall[0, 33] = np.std(D_Major_Minor_list)

    # print table
    header = ['Gen', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'Euclidean Length(mm)', 'std',
              'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std','Minor Angle','std','Major Angle','std', 'LLparent', 'std', 'LminLparent', 'std', 'LmajLparent', 'std', 'LminLmaj', 'std', 'DDparent', 'std','DminDparent', 'std','DmajDparent', 'std','DminDmaj', 'std']

    print(tabulate(values_overall, headers=header))
    print('\n')

    return np.concatenate((values_by_gen, values_overall),0)

####
# Function: find statistics on branching tree (by Strahler order) and display as table
# Inputs: branchGeom - contains properties (length, radius etc.) by Strahler branch
#         geom - contains various element properties (length, radius etc.) by element
#         orders - contains strahler order and generation of each element
# Outputs: table of information according to order and other information printed to screen
######

def summary_statistics(branchGeom, geom, orders, major_minor_results):

    # branch inputs
    branchDiam = 2 * branchGeom['radii']
    branchLen = branchGeom['length']
    branchEucLen = branchGeom['euclidean length']
    branchOrder = branchGeom['order']
    branchAngles = branchGeom['branch_angles']
    branchLenRatio = branchGeom['length ratio']
    branchDiamRatio = branchGeom['diam ratio']

    # statisitcs by order
    num_orders = int(max(branchOrder))
    values_by_order = np.zeros([num_orders, 20])

    for n_ord in range(0, num_orders):

        branch_list = (branchOrder == n_ord + 1)

        diam_list = np.extract(branch_list, branchDiam)
        len_list = np.extract(branch_list, branchLen)

        # account for zero diameters
        diam_bool = diam_list > 0
        len_bool = len_list > 0
        list = np.logical_and(diam_bool, len_bool)
        diam_list = diam_list[list]
        len_list = len_list[list]

        # assign stats for each order
        values_by_order[n_ord, 0] = n_ord + 1  # order
        values_by_order[n_ord, 1] = len(np.extract(branch_list, branch_list))  # number of branches

        values_by_order[n_ord, 2] = np.mean(np.extract(branch_list, branchLen))  # length
        values_by_order[n_ord, 3] = np.std(np.extract(branch_list, branchLen))  # length std

        values_by_order[n_ord, 4] = np.mean(diam_list)  # diameter
        values_by_order[n_ord, 5] = np.std(diam_list)  # diameter std

        values_by_order[n_ord, 6] = np.mean(np.extract(branch_list, branchEucLen))  # euclidean length
        values_by_order[n_ord, 7] = np.std(np.extract(branch_list, branchEucLen))  # euclidean length std

        values_by_order[n_ord, 8] = np.mean(len_list / diam_list)  # length / diameter
        values_by_order[n_ord, 9] = np.std(len_list / diam_list)  # length / diameter std

        values_by_order[n_ord, 10] = np.mean(
            np.extract(branch_list, branchLen) / np.extract(branch_list, branchEucLen))  # tortuosity
        values_by_order[n_ord, 11] = np.std(
            np.extract(branch_list, branchLen) / np.extract(branch_list, branchEucLen))  # tortuosity


        if n_ord < num_orders - 1:


            angle_list = np.extract(branch_list, branchAngles)
            angle_list = angle_list[angle_list > 0]

            values_by_order[n_ord, 12] = np.mean(angle_list)  # angles
            values_by_order[n_ord, 13] = np.std(angle_list)  # angles std

            lengthRatio = np.extract(branch_list, branchLenRatio)
            lengthRatio = lengthRatio[lengthRatio > 0]

            values_by_order[n_ord, 14] = np.mean(lengthRatio)  # len ratio
            values_by_order[n_ord, 15] = np.std(lengthRatio)  # len ratio

            diamRatio = np.extract(branch_list, branchDiamRatio)
            diamRatio = diamRatio[diamRatio > 0]

            values_by_order[n_ord, 16] = np.mean(diamRatio)  # diam ratio
            values_by_order[n_ord, 17] = np.std(diamRatio)  # diam std

        values_by_order[n_ord, 18] = values_by_order[n_ord-1, 1]/values_by_order[n_ord, 1]  # Bifurcation ratio
        values_by_order[n_ord, 19] = np.sum(np.square(diam_list)*np.pi/4 ) # Total CSA

    # print table
    header = ['Order', 'NumBranches', 'Length(mm)', 'std', 'Diameter(mm)', 'std', 'Euclidean Length(mm)', 'std',
              'Len/Diam', 'std', 'Tortuosity', 'std', 'Angles', 'std', 'LenRatio', 'std', 'DiamRatio', 'std','Bifurcation Ratio','TotalCSA']
    print('\n')
    print('Statistics By Order: ')
    print('..................')
    print(tabulate(values_by_order, headers=header))

    # statistics independent of order
    values_overall = np.zeros([1, 20])

    branch_list = (branchOrder > 0)
    diam_list = np.extract(branch_list, branchDiam)

    len_list = np.extract(branch_list, branchLen)
    len_list = len_list[diam_list > 0]
    diam_list = diam_list[diam_list > 0]

    angle_list = np.extract(branch_list, branchAngles)
    angle_list = angle_list[angle_list > 0]

    # assign stats for each order
    values_overall[0, 0] = -1
    values_overall[0, 1] = len(np.extract(branch_list, branch_list))  # number of branches

    values_overall[0, 2] = np.mean(len_list)  # length
    values_overall[0, 3] = np.std(len_list)  # length std

    values_overall[0, 4] = np.mean(diam_list)  # diameter
    values_overall[0, 5] = np.std(diam_list)  # diameter std

    values_overall[0, 6] = np.mean(np.extract(branch_list, branchEucLen))  # euclidean length
    values_overall[0, 7] = np.std(np.extract(branch_list, branchEucLen))  # euclidean length std

    values_overall[0, 8] = np.mean(len_list / diam_list)  # length / diameter
    values_overall[0, 9] = np.std(len_list / diam_list)  # length / diameter std

    values_overall[0, 10] = np.mean(
        np.extract(branch_list, branchLen) / np.extract(branch_list, branchEucLen))  # tortuosity
    values_overall[0, 11] = np.std(
        np.extract(branch_list, branchLen) / np.extract(branch_list, branchEucLen))  # tortuosity

    values_overall[0, 12] = np.mean(angle_list)  # angles
    values_overall[0, 13] = np.std(angle_list)  # angles std

    lengthRatio = np.extract(branch_list, branchLenRatio)
    lengthRatio = lengthRatio[lengthRatio > 0]
    values_overall[0, 14] = np.mean(lengthRatio)  # len ratio
    values_overall[0, 15] = np.std(lengthRatio)  # len ratio

    diamRatio = np.extract(branch_list, branchDiamRatio)
    diamRatio = diamRatio[diamRatio > 0]
    values_overall[0, 16] = np.mean(diamRatio)  # diam ratio
    values_overall[0, 17] = np.std(diamRatio)  # diam std

    values_overall[0, 18] = np.mean(values_by_order[1:num_orders, 18])  # Bifurcation ratio



    # print table
    header = ['     ', '           ', '          ', '   ', '            ', '   ', '                     ', '   ',
              '        ', '   ', '           ', '   ', '      ', '   ', '        ', '   ', '         ', '   ','                 ']

    print(tabulate(values_overall, headers=header))
    print('\n')

    # unpack inputs
    strahler = orders['strahler']
    generation = orders['generation']

    diam = 2*geom['radii']
    length = geom['length']
    length2 = length[(diam > 0)]
    diam = diam[(diam > 0)]
    euclid_length = geom['euclidean length']

    angles = geom['branch angles']
    angles = angles[angles > 0]  # get rid of first elem

    diam_ratio = geom['diam_ratio']
    diam_ratio = diam_ratio[(diam_ratio > 0)]

    length_ratio = geom['length_ratio']
    length_ratio = length_ratio[(length_ratio > 0)]

    # unpack inputs
    Minor_angle = major_minor_results['Minor_angle']
    Minor_angle = Minor_angle[Minor_angle > 0]

    Major_angle = major_minor_results['Major_angle']
    Major_angle = Major_angle[Major_angle > 0]

    D_Major_Minor = major_minor_results['D_maj_min']
    D_Major_Minor = D_Major_Minor[D_Major_Minor > 0]

    D_min_parent = major_minor_results['D_min_P']
    D_min_parent = D_min_parent[(D_min_parent > 0)]

    D_maj_parent = major_minor_results['D_maj_P']
    D_maj_parent = D_maj_parent[(D_maj_parent > 0)]

    L_Major_Minor = major_minor_results['L_maj_min']
    L_Major_Minor = L_Major_Minor[L_Major_Minor > 0]

    L_min_parent = major_minor_results['L_min_P']
    L_min_parent = L_min_parent[(L_min_parent > 0)]

    L_maj_parent = major_minor_results['L_maj_P']
    L_maj_parent = L_maj_parent[(L_maj_parent > 0)]

    # Segment statistics
    print('Segment statistics: ')
    print('..................')
    print('Num Segments = ' + str(len(strahler)))
    print('Total length = ' + str(np.sum(branchGeom['length'])) + ' mm')
    print('Num generations = ' + str(max(generation)))
    terminalGen = generation[(strahler == 1)]
    print('Average Terminal generation (std) = ' + str(np.mean(terminalGen)) + ' (' + str(np.std(terminalGen)) + ')')
    print('Segment Tortuosity = ' + str(np.mean(length / euclid_length)) + ' (' + str(
        np.std(length / euclid_length)) + ')')
    print('Average Length (std) = ' + str(np.mean(length)) + ' (' + str(np.std(length)) + ')')
    print('Average Euclidean Length (std) = ' + str(np.mean(euclid_length)) + ' (' + str(np.std(euclid_length)) + ')')
    print('Average Diameter (std) = ' + str(np.mean(diam)) + ' (' + str(np.std(diam)) + ')')
    print('Average L/D (std) = ' + str(np.mean(length2/diam)) + ' (' + str(np.std(length2/diam)) + ')') ########

    print('Segment Angles = ' + str(np.mean(angles)) + ' (' + str(np.std(angles)) + ')')
    print('    Minor Angle = ' + str(np.mean(Minor_angle)) + ' (' + str(np.std(Minor_angle)) + ')')
    print('    Major Angle = ' + str(np.mean(Major_angle)) + ' (' + str(np.std(Major_angle)) + ')')
    print('D/Dparent = ' + str(np.mean(diam_ratio)) + ' (' + str(np.std(diam_ratio)) + ')')
    print('    Dmin/Dparent = ' + str(np.mean(D_min_parent)) + ' (' + str(np.std(D_min_parent)) + ')')
    print('    Dmaj/Dparent = ' + str(np.mean(D_maj_parent)) + ' (' + str(np.std(D_maj_parent)) + ')')
    print('    Dmaj/Dmin = ' + str(np.mean(D_Major_Minor)) + ' (' + str(np.std(D_Major_Minor)) + ')')
    print('L/Lparent = ' + str(np.mean(length_ratio)) + ' (' + str(np.std(length_ratio)) + ')')
    print('    Lmin/Lparent = ' + str(np.mean(L_min_parent)) + ' (' + str(np.std(L_min_parent)) + ')')
    print('    Lmaj/Lparent = ' + str(np.mean(L_maj_parent)) + ' (' + str(np.std(L_maj_parent)) + ')')
    print('    Lmaj/Lmin = ' + str(np.mean(L_Major_Minor)) + ' (' + str(np.std(L_Major_Minor)) + ')')
    print('\n')

    # Find  Strahler Ratios: Rb, Rl, Rd
    Num_Branches = values_by_order[:, 1]
    Diameter_strahler = values_by_order[:, 4]
    Length_strahler = values_by_order[:, 2]
    Orders_strahler = values_by_order[:, 0]

    [Rb, r2] = find_strahler_ratio(Orders_strahler, Num_Branches)
    print('Rb = ' + str(Rb) + ' Rsq = ' + str(r2))
    [Rd, r2] = find_strahler_ratio(Orders_strahler, Diameter_strahler)
    print('Rd = ' + str(Rd) + ' Rsq = ' + str(r2))
    [Rl, r2] = find_strahler_ratio(Orders_strahler, Length_strahler)
    print('Rl = ' + str(Rl) + ' Rsq = ' + str(r2))

    return np.concatenate((values_by_order, values_overall),0)

#######################
# Function: Finds diameter scaling coefficent and creates a plot of diameters
# Inputs:  diam - an N x 1 array containing the diameter of all N branches
#          cutoff - the diameter below which will be excluded when calculating the diameter scaling coefficient
# Outputs: grad - the diameter scaling coefficient
#######################

def diam_log_cdf(diam, cutoff):

    plt.figure()

    # Reversed cumulative histogram.
    plt.subplot(1, 2, 1)
    plt.title('Reverse CDF')
    plt.ylabel('Proportion of Segments')
    plt.xlabel('Diameter')
    n_bins = np.int(np.round(len(diam) / 25)) # bin spacing
    n, bins, patches = plt.hist(diam, n_bins, density=True, histtype='step', cumulative=-1, label='Reverse cdf') # cdf

    # Add cutoff.
    plt.plot([cutoff,cutoff], [0,1],label ='cutoff')
    plt.legend()

    # Log Log plot
    plt.subplot(1, 2, 2)
    plt.ylabel('log(Number Segments)')
    plt.xlabel('log(Diameter)')

    # only take bins above cutoff
    bins=bins[0:len(bins)-1] # so same size as n
    n=n[bins>cutoff]
    bins=bins[bins>cutoff]

    # log log plot
    x=np.log(bins)
    yData=np.log(n)
    plt.plot(x,yData , 'k--', linewidth=1.5, label='Data')

    # fit line to data
    xFit=np.unique(x)
    yFit=np.poly1d(np.polyfit(x, yData, 1))(np.unique(x))
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, yData, 1))(np.unique(x)),label='linear fit')

    # Scaling Coefficient is gradient
    grad=(yFit[len(yFit)-1]-yFit[0])/(xFit[len(xFit)-1]-xFit[0])
    heading=('Diam Coeff = ' + str(grad))
    plt.title(heading)
    plt.legend()
    plt.show()

    # R^2 value
    yMean = [mean(yData) for y in yData]
    r2=1 - (sum((yFit - yData) * (yFit - yData))/sum((yMean-yData) * (yMean - yData)))
    print('Diameter Scaling Coefficient = ' + str(grad) + ' Rsquared = ' + str(r2))
    return grad, r2