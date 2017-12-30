def compute_cost(Z3, Y, parameters, lambd):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    import tensorflow as tf

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    # I have added regularization here
    regularizer = tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"])
    cost = tf.reduce_mean(cost + lambd * regularizer)

    return cost