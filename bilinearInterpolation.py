def bi(R, G, B):
    """
    params: R, G, B are the color channels of the raw image.
    e.g. R has values only for red pixels and zeros elsewhere.
    output: newR, newG, newB are interpolated color channels, 
    the resulting image is the stack of them.
    """
    import numpy as np
    import copy 
    
    newR = np. pad(copy.deepcopy(R), (1, 1), 'constant')
    newG = np.pad(copy.deepcopy(G), (1, 1), 'constant')
    newB = np.pad(copy.deepcopy(B), (1, 1), 'constant')
    
    newR[2:newR.shape[0]-1:2, 1:newR.shape[1]-2:2] = \
        (newR[2:newR.shape[0]-1:2, 0:newR.shape[1]-3:2] + \
         newR[2:newR.shape[0]-1:2, 2:newR.shape[1]-1:2]) / 2
         
    newR[1:newR.shape[0]-2:2, 2:newR.shape[1]-1:2] = \
        (newR[0:newR.shape[0]-3:2, 2:newR.shape[1]-1:2] + \
         newR[2:newR.shape[0]-1:2, 2:newR.shape[1]-1:2]) / 2
    
    newR[1:newR.shape[0]-2:2, 1:newR.shape[1]-2:2] = \
        (newR[0:newR.shape[0]-3:2, 0:newR.shape[1]-3:2] + \
         newR[0:newR.shape[0]-3:2, 2:newR.shape[1]-1:2] + \
         newR[2:newR.shape[0]-1:2, 0:newR.shape[1]-3:2] + \
         newR[2:newR.shape[0]-1:2, 2:newR.shape[1]-1:2]) / 4
         
    newG[1:newG.shape[0]-2:2, 1:newG.shape[1]-2:2] = \
        (newG[1:newR.shape[0]-2:2, 0:newG.shape[1]-3:2] + \
         newG[0:newR.shape[0]-3:2, 1:newG.shape[1]-2:2] + \
         newG[1:newR.shape[0]-2:2, 2:newG.shape[1]-1:2] + \
         newG[2:newR.shape[0]-1:2, 1:newG.shape[1]-2:2]) / 4
    
    newG[2:newG.shape[0]-1:2, 2:newG.shape[1]-1:2] = \
        (newG[2:newR.shape[0]-1:2, 1:newG.shape[1]-2:2] + \
         newG[1:newR.shape[0]-2:2, 2:newG.shape[1]-1:2] + \
         newG[2:newR.shape[0]-1:2, 3:newG.shape[1]:2] + \
         newG[3:newR.shape[0]:2, 2:newG.shape[1]-1:2]) / 4
    
    newB[1:newB.shape[0]-2:2, 2:newB.shape[1]-1:2] = \
        (newB[1:newB.shape[0]-2:2, 1:newB.shape[1]-2:2] + \
         newB[1:newB.shape[0]-2:2, 3:newB.shape[1]:2]) / 2
         
    newB[2:newB.shape[0]-1:2, 1:newB.shape[1]-2:2] = \
        (newB[1:newB.shape[0]-2:2, 1:newB.shape[1]-2:2] + \
         newB[3:newB.shape[0]:2, 1:newB.shape[1]-2:2]) / 2
         
    newB[2:newB.shape[0]-1:2, 2:newB.shape[1]-1:2] = \
        (newB[1:newB.shape[0]-2:2, 1:newB.shape[1]-2:2] + \
         newB[1:newB.shape[0]-2:2, 3:newB.shape[1]:2] + \
         newB[3:newB.shape[0]:2, 1:newB.shape[1]-2:2] + \
         newB[3:newB.shape[0]:2, 3:newB.shape[1]:2]) / 4
    
    interpolatedIm = np.stack((newR[1:-1, 1:-1], newG[1:-1, 1:-1], \
                               newB[1:-1, 1:-1]), axis=-1)
    interpolatedIm = 255 * interpolatedIm # Now scale by 255
    interpolatedIm = interpolatedIm.astype(np.uint8)
    
    return [newR, newG, newR, interpolatedIm]