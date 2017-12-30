def relu(z):
    """
    Implements relu nonlinearity
    Argument :
    Input: z
    Returns :
    g -- output
    cache -- list containing value of z
    """
    import numpy as np
    g=np.maximum(0,z)
    assert (g.shape == z.shape)
    cache=z
    return g,cache