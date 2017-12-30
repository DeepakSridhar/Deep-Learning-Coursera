import numpy as np
from create_mask import create_mask_from_window
from distribute_value import distribute_value
def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """


    # Retrieve information from cache
    (A_prev, hparameters) = (cache[0], cache[1])

    # Retrieve hyperparameters from "hparameters"
    stride = hparameters["stride"]
    f = hparameters["f"]

    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):  # loop over the training examples

        # select training example from A_prev
        a_prev = A_prev[i, :, :, :]

        for h in range(n_H):  # loop on the vertical axis
            for w in range(n_W):  # loop on the horizontal axis
                for c in range(n_C):  # loop over the channels (depth)

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = (h * stride) + f
                    horiz_start = w * stride
                    horiz_end = (w * stride) + f

                    # Compute the backward propagation in both modes.
                    if mode == "max":

                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                    #                         print(dA[i,h,w,c])

                    elif mode == "average":

                        # Get the value a from dA
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)


    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev