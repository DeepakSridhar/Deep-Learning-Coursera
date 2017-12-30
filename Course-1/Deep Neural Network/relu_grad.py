def relu_gradient(z):
    """
    Implements the gradient of relu nonlinearity
    Argument :
    Input: z
    Returns :
    g -- gradient output
    """
    import numpy as np
    k=np.nonzero(z)[0]
    g=np.zeros(z.shape)
    g[k]=1
    return g