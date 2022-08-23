import numpy as np

######
# Function: Find radius by normal projection
# Inputs: euclid_radii - radii according to shortest euclidean distance, Ne x 1
#         SkeletonImage - a logical matrix with skeleton image, Nx x Ny x Nz
#         VolumeImage - a logical matrix with volume image, Nx x Ny x Nz
#         elems - and Ne x 3 araay with elems, must be in Strahler order
#         nodes - Nn x 3 array with node coordinates (it is assumed all nodes are far from the edges of the image) - [Y, X, Z]
# Outputs: normal_radii - radii according to normal projections, Ne x 1
#######

def find_radius_normal_projection(SkeletonImage, VolumeImage, elems, nodes, euclid_radii):

    # switch nodes to agree with images
    placeHolder=np.copy(nodes[:,0])
    nodes[:, 0]=nodes[:,1]
    nodes[:, 1]=placeHolder

    NumElems = len(elems)
    normal_radii=np.zeros((NumElems))
    normal_radii_std=np.zeros((NumElems)) # within element variation in radius, this isn't needed but has been included just in case

    # Starts and terminal elements and works back up the tree through all the image voxels
    totalErrors=0

    for ne in range(NumElems-1, -1, -1):
        #print(ne)
        coord = np.squeeze(nodes[int(elems[ne, 2]),:]) # start of element
        endCoord = np.squeeze(nodes[int(elems[ne, 1]),:]) # end of element

        count = 0
        elementVoxelSet=np.zeros((1,3))
        elementVoxelSet[count,:]=coord
        errorForElement = 0

        while (~np.prod(coord == endCoord)) & (count < 1000): # ie not another junction voxel + arbitrary absolute check

            # find next coord
            x_start=(int(coord[0]) - 2)
            y_start=int(coord[1] - 2)
            z_start=int(coord[2] - 2)
            large_neighbourhood = np.copy(SkeletonImage[x_start:x_start+5, y_start:y_start+5, z_start:z_start+5]) # 5 x 5 x 5 region around currect coord
            # find next coord
            [nextCoord, error] = find_connected_voxels(large_neighbourhood, coord, endCoord)

            # update image
            SkeletonImage[int(coord[0]), int(coord[1]), int(coord[2])] = 0 # get rid of coordinates once they have been used

            if error:
                errorForElement=1 # will get an error if there is an error at any point in branch

            # update loop
            coord = nextCoord # move up element
            elementVoxelSet=np.append(elementVoxelSet, np.reshape(coord,[1,3]), axis=0)

        SkeletonImage[int(endCoord[0]), int(endCoord[1]), int(endCoord[2])] = 1 # keep the junction voxel as may encounter again

        if count >= 1000:
            print('stuck in loop error') # such as by jumping from one branch to another
            errorForElement = 1

        if errorForElement:
            totalErrors=totalErrors+1
            #shape=np.shape(elementVoxelSet)
            #for k in range (0, shape[0]):
            #    coord2=np.squeeze(elementVoxelSet[k,:])
            #    SkeletonImage[int(coord2[0]),int(coord2[1]),int(coord2[2])]=1 #idea was to return unsuccessfully tracked branches but didnt really work


        # only need to keep inner third of the element for radius calculations
        branchSize = np.shape(elementVoxelSet)
        branchSize=branchSize[0]
        elementVoxelSet = elementVoxelSet[int(np.ceil(branchSize / 3)):int(np.ceil(branchSize - branchSize / 3)), :]
        numVoxels = np.shape(elementVoxelSet)
        numVoxels=numVoxels[0]
        gap = 2 # determines how far we look ahead to get centre line direction (also determined how discretized angles are)

        # Estimate radius by normal projection
        if (numVoxels > gap)&(errorForElement == 0): # can go on to calculate radii

            distanceSet = np.zeros((numVoxels - gap, 1))

            for i in range(0,numVoxels - gap):

                coord1 = np.squeeze(elementVoxelSet[i,:])
                coord2 = np.squeeze(elementVoxelSet[i + gap,:])
                distanceSet[i] = np.mean(find_distances_using_normal(coord1, coord2, VolumeImage))

            # Find mean
            normal_radii[ne] = np.mean(distanceSet)
            normal_radii_std[ne] = np.std(distanceSet)
        else:
            normal_radii[ne] =-1
            normal_radii_std[ne] = -1

    print('Number of Elements that could not successfully be tracked: '+str(totalErrors))

    # Compare radii to euclidean radii
    normal_radii[normal_radii < 0] = euclid_radii[normal_radii<0]
    euclid_radii[euclid_radii == 0] = normal_radii[euclid_radii == 0]
    euclid_radii[euclid_radii == 0]=np.min(euclid_radii[euclid_radii>0]) # so no chance of div 0
    difference = abs(normal_radii - euclid_radii)/ euclid_radii
    cutoff = 0.33333 # distances that are larger than cutoff are not used, 1 means that the distance is the same magnitude as the euclidean distance
    normal_radii[difference > cutoff] = euclid_radii[difference > cutoff]

    return (normal_radii)

######
# Function: Find radius for a single point on an element using normal projects
# Inputs: coord1, coord2 - 3d coordinates of two points on the centreline of the element
#         VolumeImage - a logical matrix with volume image, Nx x Ny x Nz
# Outputs: distances - an M x 1 array of various distances estimated at this slice
#######

def find_distances_using_normal(coord1, coord2, VolumeImage):

    # get centre line vector
    centre = np.double((coord1 - coord2))/ np.linalg.norm(np.double(coord1 - coord2))

    numSamples = 10
    distances = np.zeros(numSamples)
    normal = np.zeros(3)

    for i in range(0,numSamples):

        # Randomly assign normal vector, using the dot product rule (centre.normal==0) and avoiding div0 errors
        if centre[2]!= 0:
            normal[0] = np.random.rand() - 0.5
            normal[1] = np.random.rand() - 0.5
            normal[2] = -(centre[0] * normal[0] + centre[1] * normal[1])/ centre[2]

        elif centre[0] != 0:
            normal[2] = np.random.rand() - 0.5
            normal[1] = np.random.rand() - 0.5
            normal[0] = -(centre[2] * normal[2] + centre[1] * normal[1]) / centre[0]

        else: # centre[1]!= 0:
            normal[0] = np.random.rand() - 0.5
            normal[2] = np.random.rand() - 0.5
            normal[1] = -(centre[0] * normal[0] + centre[2] * normal[2]) / centre[1]

        normal = normal / np.linalg.norm(normal)

        # Find distances
        step = 0
        counter = 0
        currentValue = 1
        while (currentValue == 1) & (counter < 1000): # check if in vessel (plus arbitrary check)

             step = step + 0.2 # step update by 1/5 of a voxel (could increase in order to speed up)
             counter = counter + 1
             currentPosition = np.double(coord1) + step*normal # take step in direction of normal vector
             currentPosition = np.round(currentPosition)
             currentValue = VolumeImage[int(currentPosition[0]), int(currentPosition[1]), int(currentPosition[2])]
        distances[i] = step - 0.2

    return distances

######
# Function: Find next voxel up the skeleton element
# Inputs: coord - current position on element
#         endCoord - final position on element
#         large neighbourhood - the 5 x 5 x 5 area surrounding coord, a section of skeleton image
# Outputs: nextCoord - next position along element
#######

def find_connected_voxels(large_neighbourhood, coord, endCoord):

    error=0 # default
    large_neighbourhood[2, 2, 2] = 0  # so can't stay at same spot
    neighbourhood = np.copy(large_neighbourhood[1:4, 1:4, 1:4])  # 3 x 3 x 3 area surrounding the coord

    inds = np.where(neighbourhood == 1) # new places to step to
    numInds=len(inds[0])
    subs = np.squeeze(np.column_stack([inds[0],inds[1],inds[2]]))

    # Case of one choice of where to step
    if numInds == 1:
        nextCoord = [coord[0] + subs[0] - 1, coord[1] + subs[1] - 1, coord[2] + subs[2] - 1] # current voxel is at [1,1,1] in neighbourhood

    # Case of no options of where to go
    else: #Either numInds==0 (nowhere to go in direct neighbour hood) OR numInds>1 (too many places to choose in direct neighbourhood

        large_inds = np.where(large_neighbourhood == 1) # places to jump to
        large_subs=np.squeeze(np.column_stack([large_inds[0],large_inds[1],large_inds[2]]))
        numLargeInds = len(large_inds[0])

        if numLargeInds == 1: # only one place to jump, go there
            nextCoord = [coord[0] + large_subs[0] - 2, coord[1] + large_subs[1] - 2, coord[2] + large_subs[2] - 2] # current voxel is at [2, 2, 2] in large neighbourhood

        elif numLargeInds > 1:  # choose where to jump using distance criteria
            endDist = np.zeros(numLargeInds)
            done = 0

            for i in range(0,numLargeInds):

                nextCoord = [coord[0] + large_subs[i,0] - 2, coord[1] + large_subs[i,1] - 2, coord[2] + large_subs[i,2] - 2]  # current voxel is at [2, 2, 2] in large neighbourhood

                # fast track to end if it is in the large neighbourhood (most of the time this section solves it)
                if (np.prod(nextCoord == endCoord)):
                    done = 1
                    break

                # distance to end coord
                endDist[i] = np.sqrt(np.sum(np.square(nextCoord - endCoord)))

            if done == 0: # choose point closest to end point
                # print('Connectivity Error')
                error = 1  # have failed to get to end coord of this branch
                nextCoord = endCoord  # fast track to end

        else: # can't go anywhere, error
            #print('Connectivity Error')
            error = 1 # have failed to get to end coord of this branch
            nextCoord = endCoord # fast track to end

    next_coord=np.array(nextCoord) #check type is correct

    return (next_coord, error)




