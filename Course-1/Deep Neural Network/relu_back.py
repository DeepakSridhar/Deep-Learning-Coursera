def relu_backward(dA, cache):
    # from relu_grad import relu_gradient
    # z=cache
    # dZ=dA*relu_gradient(z)
    import  numpy as np
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return  dZ

