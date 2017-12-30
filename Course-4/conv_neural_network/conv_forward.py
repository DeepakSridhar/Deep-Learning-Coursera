from zero_pad import zero_pad
from conv_single_step import conv_single_step
import numpy as np

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = (A_prev.shape[0], A_prev.shape[1], A_prev.shape[2], A_prev.shape[3])

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = (W.shape[0], W.shape[1], W.shape[2], W.shape[3])

    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor.
    n_H = int((n_H_prev + (2 * pad) - f) / stride + 1)
    n_W = int((n_W_prev + (2 * pad) - f) / stride + 1)
    #     print(n_H)
    #     print(n_W)

    # Initialize the output volume Z with zeros.
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    l = 1
    for i in range(0, m):  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]  # Select ith training example's padded activation
        for h in range(0, n_H):  # loop over vertical axis of the output volume
            for w in range(0, n_W):  # loop over horizontal axis of the output volume
                for c in range(0, n_C):  # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = f + (h * stride)
                    horiz_start = w * stride
                    horiz_end = f + (w * stride)

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell).
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])


    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache