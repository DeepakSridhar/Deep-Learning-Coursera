def sigmoid_gradient(z):
    """
    Implements gradient of sigmoid function
    Argument :
    Input: z
    Returns :
    g -- output
    cache -- list containing value of z
    """
    from sigmoid import sigmoid
    a=sigmoid(z)[0]
    g=a*(1-a)
    return g