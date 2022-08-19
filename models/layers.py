import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import ReLU, Softmax
from tensorflow.keras.activations import tanh

from .dynamic_routing import KDE_routing, squash

class AttentionWeighter(layers.Layer):
    """
    Attention-Weight subclassed-keras.layers.Layer for convolutional layer
    based on https://arxiv.org/pdf/1803.07179.pdf and https://arxiv.org/abs/1903.11748
    """
    def __init__(self, 
                 n_filters,  
                 kernel_initializer=GlorotNormal(seed=None), 
                 **kwargs):
        """

        Args:
            n_filters (int): number of filters in the input data
            kernel_initializer (tf.keras.initializers, optional): _description_. Defaults to GlorotNormal(seed=None).
            weight initialization
        """
        
        super(AttentionWeighter, self).__init__(**kwargs)
    
        self.kernel_init = kernel_initializer
        self.n_filters = n_filters

    def build(self, input_shape):
        """ 
        Initializes the shape of the layer, including that of weights and bias
        """
        self.in_filers = input_shape[-1]
        self.length = input_shape[-2]
        self.W = self.add_weight(
            shape=(self.n_filters, 1),
            initializer=self.kernel_init,
            dtype=tf.float32,
            name="W",
            trainable=True
        )
            
    def call(self, h_i):
        
        # Computes the output of convolutions through an attention-based weight
        alpha = Softmax()(tanh(tf.einsum('ij, kjm -> kim', tf.transpose(self.W), tf.transpose(h_i,perm=[0,2,1]))))
        # Combination computation
        gamma = ReLU()(tf.einsum('ijk,ijm->imk', h_i, tf.transpose(alpha, perm=[0,2,1])))
   
        return gamma
    




class FlattenCaps(layers.Layer):
    """
    Layer class to flatten the last two dimensions of squashed and normalized capsules
    """
    def __init__(self):
        super(FlattenCaps, self).__init__()
    
    def call(self, poses, activations):
        """
    
        Args:
            poses (tf.tensor): squashed caps
            activations (tf.tensor): normalized caps

        Returns:
            poses and activations reshaped in last dimension
        """
        b,_,k,l,m = poses.shape
        poses = tf.reshape(poses, (b,k*l*m, -1))
        b,j,k,l,_ = activations.shape
        activations = tf.reshape(activations, (b,j*k*l, -1))
        return poses, activations
    

class PrimaryCaps(layers.Layer):
    """
    Layer to extract the primary capsules based on 
    https://proceedings.neurips.cc/paper/2017/file/2cad8fa47bbef282badbb8de5374b894-Paper.pdf

    Args:
        num_capsules (int): number of capsules to create
        out_filters (int): output shape of filters
        kernel_size (int): size of convolutinal kernel. Since we are dealing with a 1-d convolution 
            this must be just an int
        stride (int): stride of convolution
        
    """
    def __init__(self, num_capsules, out_filters, kernel_size, stride, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)

        # instantiating capsule extractor
        self.capsules = layers.Conv1D(out_filters * num_capsules, 
                                      kernel_size, stride, 
                                      kernel_initializer="glorot_uniform")


        self.out_filters = out_filters
        self.num_capsules = num_capsules


    def call(self, x):
        batch_size = x.shape[0]
        # Computing capsules
        u = tf.reshape(self.capsules(u), (batch_size, self.num_capsules, self.out_filters, -1,1))
        # Decreasing values from 0 to 1 range
        poses = squash(u, axis=1)
        # generating activations from squashed vector
        activations = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(poses), axis=1))
        
        return poses, activations
    
class FCCaps(layers.Layer):
    """"
    Forward-capsule layer that performs kernel-density based dynamic routing

    Args:
        output_capsule_num (int): number of output capsules
        input_capsule_num (int): capsule input dimension
        in_filters (int): number of channels in input
        out_filters (int): number of output filters
    """
    def __init__(self,  output_capsule_num, input_capsule_num, in_filters, out_filters):
        super(FCCaps, self).__init__()

        self.out_filters = out_filters
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num

        # Initializing weight and biases
        self.w1 = self.add_weight(
                                shape=(self.input_capsule_num, self.output_capsule_num, self.out_filters, in_filters), 
                                initializer="glorot_uniform", 
                                trainable=True)
        self.b = self.add_weight(
                                shape=(self.input_capsule_num, self.output_capsule_num, 1, 1), 
                                initializer="zeros", 
                                trainable=True)

    def call(self, x):

        x = tf.expand_dims(x, axis=2)
        # batch multiplication estimation of weighted combination capsules
        u_hat = tf.einsum('ijkl, jclm -> ijckm ',self.W1, x)
        poses, activations = KDE_routing(self.b, u_hat)

        return poses, activations
