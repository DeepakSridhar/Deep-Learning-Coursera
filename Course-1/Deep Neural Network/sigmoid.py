def sigmoid(z):
    """
    Implements sigmoid activation function
    Argument :
    Input: z
    Returns :
    g -- output
    cache -- list containing value of z
    """
    import numpy as np
    g=1/(1+np.exp(-z))
    cache=z
    return g,cache