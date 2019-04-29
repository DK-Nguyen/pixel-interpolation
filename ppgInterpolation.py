def hueTransit(l1, l2, l3, v1, v3):
    if (l1 < l2 and l2 < l3) or (l1 > l2 and l2 > l3):
        return v1 + (v3-v1) * (l2-l1)/(l3-l1)
    else:
        return (v1+v3)/2 + (l2*2 - l1 - l3)/4

def ppg(R, G, B):
    """
    params: R, G, B are the color channels of the raw image.
    e.g. R has values only for red pixels and zeros elsewhere.
    output: newR, newG, newB are interpolated color channels, 
    the resulting image is the stack of them.
    """
    import numpy as np
    import copy 
    
    newR = np.pad(copy.deepcopy(R), (2, 2), 'constant')
    newG = np.pad(copy.deepcopy(G), (2, 2), 'constant')
    newB = np.pad(copy.deepcopy(B), (2, 2), 'constant')
    
    
    # PHASE 1.1: find the Green Values for the Blue pixels
    deltaN = np.zeros(newB.shape)
    deltaN[2:deltaN.shape[0]-3:2, 2:deltaN.shape[1]-3:2] = \
            np.absolute(newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
                    newB[0:newB.shape[0]-5:2, 2:newB.shape[1]-3:2]) * 2 + \
            np.absolute(newG[1:newG.shape[0]-4:2, 2:newG.shape[1]-3:2] - \
                        newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2])       
    
    deltaE = np.zeros(newB.shape)
    deltaE[2:deltaE.shape[0]-3:2, 2:deltaE.shape[1]-3:2] = \
            np.absolute(newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
                    newB[2:newB.shape[0]-3:2, 4:newB.shape[1]-1:2]) * 2 + \
            np.absolute(newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] - \
                        newG[2:newG.shape[0]-3:2, 1:newG.shape[1]-4:2])
                    
    deltaW = np.zeros(newB.shape)
    deltaW[2:deltaW.shape[0]-3:2, 2:deltaW.shape[1]-3:2] = \
            np.absolute(newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
                    newB[2:newB.shape[0]-3:2, 0:newB.shape[1]-5:2]) * 2 + \
            np.absolute(newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] - \
                        newG[2:newG.shape[0]-3:2, 1:newG.shape[1]-4:2])
            
    deltaS = np.zeros(newB.shape)
    deltaS[2:deltaS.shape[0]-3:2, 2:deltaS.shape[1]-3:2] = \
            np.absolute(newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
                    newB[4:newB.shape[0]-1:2, 2:newB.shape[1]-3:2]) * 2 + \
            np.absolute(newG[1:newG.shape[0]-4:2, 2:newG.shape[1]-3:2] - \
                        newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2])
    
    deltas = np.stack((deltaN, deltaE, deltaW, deltaS), axis=-1)
    minIndex = np.argmin(deltas, axis=-1)
    
    deltaN_smallest = np.zeros(newB.shape)
    deltaN_smallest[2:deltaN_smallest.shape[0]-3:2, 2:deltaN_smallest.shape[1]-3:2] = \
          (newG[1:newG.shape[0]-4:2, 2:newG.shape[1]-3:2] * 3 + \
           newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2] + \
           newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
           newB[0:newB.shape[0]-5:2, 2:newB.shape[1]-3:2]) / 4          
    
    deltaE_smallest = np.zeros(newB.shape)
    deltaE_smallest[2:deltaE_smallest.shape[0]-3:2, 2:deltaE_smallest.shape[1]-3:2] = \
          (newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] * 3 + \
           newG[2:newG.shape[0]-3:2, 1:newG.shape[1]-4:2] + \
           newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
           newB[2:newB.shape[0]-3:2, 4:newB.shape[1]-1:2]) / 4 
    
    deltaW_smallest = np.zeros(newB.shape)
    deltaW_smallest[2:deltaW_smallest.shape[0]-3:2, 2:deltaW_smallest.shape[1]-3:2] = \
          (newG[2:newG.shape[0]-3:2, 1:newG.shape[1]-4:2] * 3 + \
           newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] + \
           newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
           newB[2:newB.shape[0]-3:2, 0:newB.shape[1]-5:2]) / 4 
    
    deltaS_smallest = np.zeros(newB.shape)
    deltaS_smallest[2:deltaS_smallest.shape[0]-3:2, 2:deltaS_smallest.shape[1]-3:2] = \
          (newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2] * 3 + \
           newG[1:newG.shape[0]-4:2, 2:newG.shape[1]-3:2] + \
           newB[2:newB.shape[0]-3:2, 2:newB.shape[1]-3:2] - \
           newB[4:newB.shape[0]-1:2, 2:newB.shape[1]-3:2]) / 4
           
    deltas_smallest = np.stack((deltaN_smallest, deltaE_smallest, \
                                deltaW_smallest, deltaS_smallest), axis=-1)
    
    GforB = np.zeros(minIndex.shape)
    for i in range(minIndex.shape[0]):
        for j in range(minIndex.shape[1]):
            GforB[i,j] = deltas_smallest[i,j,minIndex[i,j]]
            
    
    # PHASE 1.2: find the Green Values for the Red pixels
    deltaN2 = np.zeros(newR.shape)
    deltaN2[3:deltaN2.shape[0]-2:2, 3:deltaN2.shape[1]-2:2] = \
            np.absolute(newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
                    newR[1:newR.shape[0]-4:2, 3:newR.shape[1]-2:2]) * 2 + \
            np.absolute(newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] - \
                        newG[4:newG.shape[0]-1:2, 3:newG.shape[1]-2:2])       
    
    deltaE2 = np.zeros(newR.shape)
    deltaE2[3:deltaE2.shape[0]-2:2, 3:deltaE2.shape[1]-2:2] = \
            np.absolute(newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
                    newR[3:newR.shape[0]-2:2, 5:newR.shape[1]:2]) * 2 + \
            np.absolute(newG[3:newG.shape[0]-2:2, 4:newG.shape[1]-1:2] - \
                        newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2])
                    
    deltaW2 = np.zeros(newR.shape)
    deltaW2[3:deltaW2.shape[0]-2:2, 3:deltaW2.shape[1]-2:2] = \
            np.absolute(newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
                    newR[3:newR.shape[0]-2:2, 1:newR.shape[1]-4:2]) * 2 + \
            np.absolute(newG[3:newG.shape[0]-2:2, 4:newG.shape[1]-1:2] - \
                        newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2])
            
    deltaS2 = np.zeros(newR.shape)
    deltaS2[3:deltaS2.shape[0]-2:2, 3:deltaS2.shape[1]-2:2] = \
            np.absolute(newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
                    newR[5:newR.shape[0]:2, 3:newR.shape[1]-2:2]) * 2 + \
            np.absolute(newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] - \
                        newG[4:newG.shape[0]-1:2, 3:newG.shape[1]-2:2])
    
    deltas2 = np.stack((deltaN2, deltaE2, deltaW2, deltaS2), axis=-1)
    minIndex2 = np.argmin(deltas2, axis=-1)
    
    deltaN_smallest2 = np.zeros(newR.shape)
    deltaN_smallest2[3:deltaN_smallest2.shape[0]-2:2, 3:deltaN_smallest2.shape[1]-2:2] = \
          (newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] * 3 + \
           newG[4:newG.shape[0]-1:2, 3:newG.shape[1]-2:2] + \
           newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
           newR[1:newR.shape[0]-4:2, 3:newR.shape[1]-2:2]) / 4          
    
    deltaE_smallest2 = np.zeros(newR.shape)
    deltaE_smallest2[3:deltaE_smallest2.shape[0]-2:2, 3:deltaE_smallest2.shape[1]-2:2] = \
          (newG[3:newG.shape[0]-2:2, 4:newG.shape[1]-1:2] * 3 + \
           newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2] + \
           newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
           newR[3:newR.shape[0]-2:2, 5:newR.shape[1]:2]) / 4 
    
    deltaW_smallest2 = np.zeros(newR.shape)
    deltaW_smallest2[3:deltaW_smallest2.shape[0]-2:2, 3:deltaW_smallest2.shape[1]-2:2] = \
          (newG[3:newG.shape[0]-2:2, 2:newG.shape[1]-3:2] * 3 + \
           newG[3:newG.shape[0]-2:2, 4:newG.shape[1]-1:2] + \
           newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
           newR[3:newR.shape[0]-2:2, 1:newR.shape[1]-4:2]) / 4 
    
    deltaS_smallest2 = np.zeros(newR.shape)
    deltaS_smallest2[3:deltaS_smallest2.shape[0]-2:2, 3:deltaS_smallest2.shape[1]-2:2] = \
          (newG[4:newG.shape[0]-1:2, 3:newG.shape[1]-2:2] * 3 + \
           newG[2:newG.shape[0]-3:2, 3:newG.shape[1]-2:2] + \
           newR[3:newR.shape[0]-2:2, 3:newR.shape[1]-2:2] - \
           newR[5:newR.shape[0]-0:2, 3:newR.shape[1]-2:2]) / 4
           
    deltas_smallest2 = np.stack((deltaN_smallest2, deltaE_smallest2, \
                                deltaW_smallest2, deltaS_smallest2), axis=-1)
    
    GforR = np.zeros(minIndex2.shape)
    for i in range(minIndex2.shape[0]):
        for j in range(minIndex2.shape[1]):
            GforR[i,j] = deltas_smallest2[i,j,minIndex2[i,j]]                
    
    interpolatedG = newG + GforB + GforR
    
    # PHASE 2.1: Computing Red Values for Green Pixels
    interpolatedR = copy.deepcopy(newR)
    # even rows
    for i in range(2, interpolatedR.shape[0]-3,2):
        for j in range(3, interpolatedR.shape[1]-2,2):
            interpolatedR[i,j] = hueTransit(interpolatedG[i-1,j], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i-1,j], \
                                            interpolatedR[i-1,j],
                                            interpolatedR[i+1,j])
    
    # odd rows
    for i in range(3, interpolatedR.shape[0]-2,2):
        for j in range(2, interpolatedR.shape[1]-3,2):
            interpolatedR[i,j] = hueTransit(interpolatedG[i,j-1], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i,j+1], \
                                            interpolatedR[i,j-1],
                                            interpolatedR[i,j+1])
            
    # PHASE 2.2: Computing Blue Values for Green Pixels
    interpolatedB = copy.deepcopy(newB)
    # even rows
    for i in range(2, interpolatedB.shape[0]-3,2):
        for j in range(3, interpolatedB.shape[1]-2,2):
            interpolatedB[i,j] = hueTransit(interpolatedG[i,j-1], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i,j+1], \
                                            interpolatedB[i,j-1],
                                            interpolatedB[i,j+1])
    # odd rows
    for i in range(3, interpolatedB.shape[0]-2,2):
        for j in range(2, interpolatedB.shape[1]-3,2):
            interpolatedB[i,j] = hueTransit(interpolatedG[i-1,j], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i-1,j], \
                                            interpolatedB[i-1,j],
                                            interpolatedB[i-1,j])
    
    # PHASE 3: Computing Red Values at Blue Pixels
    for i in range(2, interpolatedR.shape[0]-3,2):
        for j in range(2, interpolatedR.shape[1]-3,2):
            deltaNE = np.absolute(interpolatedR[i-1, j+1] - interpolatedR[i+1, j-1]) + \
                      np.absolute(interpolatedB[i-2, j+2] - interpolatedB[i, j]) + \
                      np.absolute(interpolatedB[i, j] - interpolatedB[i+2, j-2]) + \
                      np.absolute(interpolatedG[i-1, j+1] - interpolatedG[i, j]) + \
                      np.absolute(interpolatedG[i, j] - interpolatedG[i+1, j-1])
                      
            deltaNW = np.absolute(interpolatedR[i-1, j-1] - interpolatedR[i+1, j+1]) + \
                      np.absolute(interpolatedB[i-2, j-2] - interpolatedB[i, j]) + \
                      np.absolute(interpolatedB[i, j] - interpolatedB[i+2, j+2]) + \
                      np.absolute(interpolatedG[i-1, j-1] - interpolatedG[i, j]) + \
                      np.absolute(interpolatedG[i, j] - interpolatedG[i+1, j+1])
                      
            if deltaNE < deltaNW:
                interpolatedR[i,j] = hueTransit(interpolatedG[i-1,j+1], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i+1,j-1], \
                                            interpolatedR[i-1,j+1],
                                            interpolatedR[i+1,j-1])
            else:
                interpolatedR[i,j] = hueTransit(interpolatedG[i-1,j-1], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i+1,j+1], \
                                            interpolatedR[i-1,j-1],
                                            interpolatedR[i+1,j+1])
                
    # PHASE 4: Computing Blue Values at Red Pixels 
    for i in range(3, interpolatedB.shape[0]-2,2):
        for j in range(3, interpolatedB.shape[1]-2,2):
            deltaNE = np.absolute(interpolatedB[i-1, j+1] - interpolatedB[i+1, j-1]) + \
                      np.absolute(interpolatedR[i-2, j+2] - interpolatedR[i, j]) + \
                      np.absolute(interpolatedR[i, j] - interpolatedR[i+2, j-2]) + \
                      np.absolute(interpolatedG[i-1, j+1] - interpolatedG[i, j]) + \
                      np.absolute(interpolatedG[i, j] - interpolatedG[i+1, j-1])
                      
            deltaNW = np.absolute(interpolatedB[i-1, j-1] - interpolatedB[i+1, j+1]) + \
                      np.absolute(interpolatedR[i-2, j-2] - interpolatedR[i, j]) + \
                      np.absolute(interpolatedR[i, j] - interpolatedR[i+2, j+2]) + \
                      np.absolute(interpolatedG[i-1, j-1] - interpolatedG[i, j]) + \
                      np.absolute(interpolatedG[i, j] - interpolatedG[i+1, j+1])
                      
            if deltaNE < deltaNW:
                interpolatedB[i,j] = hueTransit(interpolatedG[i-1,j+1], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i+1,j-1], \
                                            interpolatedB[i-1,j+1],
                                            interpolatedB[i+1,j-1])
            else:
                interpolatedB[i,j] = hueTransit(interpolatedG[i-1,j-1], \
                                            interpolatedG[i,j],   \
                                            interpolatedG[i+1,j+1], \
                                            interpolatedB[i-1,j-1],
                                            interpolatedB[i+1,j+1])
    
    interpolatedIm = np.stack((interpolatedR[2:-2, 2:-2], interpolatedG[2:-2, 2:-2], \
                               interpolatedB[2:-2, 2:-2]), axis=-1)
    interpolatedIm = 255 * interpolatedIm # Now scale by 255
    interpolatedIm = interpolatedIm.astype(np.uint8)
    
    return interpolatedR, interpolatedG, interpolatedB, interpolatedIm

