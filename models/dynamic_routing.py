import tensorflow as tf
from tensorflow.keras import backend as K

def dynamic_routing(b_ij, u_hat, input_capsule_num, num_iters=2):
    """
    Standard dynamic routing algorithm from Hinton et al. paper
    
    ********* in development ******************************
    """

    for i in range(num_iters):
        leak = tf.math.reduce_sum(tf.zeros_like(b_ij), axis=2, keepdims=True)
        leaky_logits = tf.concat((leak, b_ij),2)
        leaky_routing = tf.nn.softmax(leaky_logits, axis=2)
        c_ij = tf.expand_dims(leaky_routing, axis=4)
        v_j = tf.math.reduce_sum(squash((c_ij * u_hat), axis=3), axis=1, keepdims=True)
        if i < num_iters - 1:
            b_ij = b_ij + tf.math.reduce_sum((tf.concat([v_j] * input_capsule_num, 1) * u_hat), axis=3)

    poses = tf.expand_dims(v_j, axis=1)
    activations = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(poses), axis=2))
    return poses, activations


def KDE_routing(b_ij, u_hat):
    """
    Layer to compute the fast Dynamic Routing Based on Weighted Kernel Density Estimation
    https://arxiv.org/abs/1805.10807
    
    Args:
        b_ij (tf.tensor): bias
        u_hat (_type_): weighted-combined capsule

    Returns:
        tf.tensor: squashed capsules
        tf.tensor: activated capsules
    """
    num_iterations = 3
    for i in range(num_iterations):
        # estimating soft-logits
        c_ij = tf.expand_dims(tf.nn.softmax(b_ij, axis=2),axis=3)
        # summing along batched axis
        c_ij = tf.math.reduce_sum(c_ij/c_ij, axis=0, keepdims=True)
        # reducing vector inputs to be close to 0 or 1
        v_j = tf.squash(tf.math.reduce_sum((c_ij * u_hat),axis=1, keepdims=True), axis=3)
        if i < num_iterations - 1:
            dd = 1 - tf.math.reduce_sum(tf.math.square((squash(u_hat, axis=3)-v_j)), axis=3)
            b_ij = b_ij + dd
    poses = tf.expand_dims(v_j, axis=1)
    activations = tf.math.sqrt(tf.math.reduce_sum((tf.math.square(poses)), axis=2))
    return poses, activations

def squash(vec, axis=-1):
    """
    
    Non-linear function from paper https://arxiv.org/abs/1710.09829?context=cs. Formula (1)
    This function squashes long vectors to 1 and smaller ones to sum up to 0

    Args:
        vec: (tf.tensor) , N-dim tensor
        axis: (int) the axis to squash
    Return: 
        tf.tensor vector of the same size of that of the input vector
    """
    s_squared_norm = K.sum(K.square(vec), axis, keepdims=True)
    scale = (s_squared_norm / (1 + s_squared_norm)) * (K.sqrt(s_squared_norm + K.epsilon()))
    return scale * vec