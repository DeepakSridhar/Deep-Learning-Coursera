def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    import numpy as np

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    # logprobs = None
    cost = -(1 / m) * (np.sum(np.dot(Y, np.log(A2).T) + np.dot(1 - Y, np.log(1 - A2).T)))

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost