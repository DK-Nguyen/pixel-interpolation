def mse(im, interpolatedIm):
    """
    params: im: the ground truth image 
            interpolatedIm: the interpolated image
    output: the Mean Squared Error of the interpolated image compared to 
    the ground truth image
    """
    import numpy as np
    mse = np.sum((im.astype(np.float32) - \
                  interpolatedIm.astype(np.float32))**2)
    mse = mse / (im.shape[0]*im.shape[1]*im.shape[2])
    return mse

def mae(im, interpolatedIm):
    """
    params: im: the ground truth image 
            interpolatedIm: the interpolated image
    output: the Mean Absolute Error of the interpolated image compared to 
    the ground truth image
    """
    import numpy as np
    mae = np.sum(np.absolute(im.astype(np.float32) - \
                             interpolatedIm.astype(np.float32)))
    mae = mae / (im.shape[0]*im.shape[1]*im.shape[2])
    return mae