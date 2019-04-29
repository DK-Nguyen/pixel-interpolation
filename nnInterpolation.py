def nearestNeighbor(R, G, B):
    """
    params: R, G, B are the color channels of the raw image.
    e.g. R has values only for red pixels and zeros elsewhere.
    output: newR, newG, newB are interpolated color channels, 
    the resulting image is the stack of them.
    """
    import copy 
    import numpy as np
    
    newR = copy.deepcopy(R)
    newG = copy.deepcopy(G)
    newB = copy.deepcopy(B)
    
    newR[0::2, :] = newR[1::2, :] 
    newR[:, 0::2] = newR[:, 1::2]
    
    newG[0::2, 0::2] = newG[0::2, 1::2]
    newG[1::2, 1::2] = newG[1::2, 0::2]
    
    newB[1::2, :] = newB[0::2, :]
    newB[:, 1::2] = newB[:, 0::2]
    
    interpolatedIm =np. stack((newR, newG, newB), axis=-1)      
    interpolatedIm = 255 * interpolatedIm # Now scale by 255
    interpolatedIm = interpolatedIm.astype(np.uint8)
    return [newR, newG, newB, interpolatedIm]


def nearestNeighborForLoop(R, G, B):
    """
    params: R, G, B are the color channels of the raw image.
    e.g. R has values only for red pixels and zeros elsewhere.
    output: newR, newG, newB are interpolated color channels, 
    the resulting image is the stack of them.
    """
    import copy 
    import numpy as np
    
    newR = copy.deepcopy(R)
    newG = copy.deepcopy(G)
    newB = copy.deepcopy(B)
    for row in range(0, R.shape[0]-1, 2):
        for col in range(0, R.shape[1]-1, 2):
            newR[row:row + 2, col:col + 2] = \
            newR[row + 2 - 1, col + 2 - 1]
            
            newG[row, col] = newG[row, col + 2-1]
            newG[row + 2 - 1, col + 2 - 1] = \
            newG[row + 2 - 1, col]
            
            newB[row:row + 2, col:col + 2] = \
            newB[row, col]
            
    interpolatedIm = np.stack((newR, newG, newB), axis=-1)
    interpolatedIm = 255 * interpolatedIm # Now scale by 255
    interpolatedIm = interpolatedIm.astype(np.uint8)
    
    return [newR, newG, newB, interpolatedIm]


