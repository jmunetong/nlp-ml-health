import tensorflow as tf
from .layers import AttentionWeighter
from .model_abstract import Model_Abstract
from tensorflow.keras.layers import Embedding, Conv1D, Concatenate, Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy

import unittest

class NNC_3(Model_Abstract):
    
    """
    Model based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9413635
    Attentional-based hierarchical model with causal convolutions. While this algorithm 
    has been meant to be implemented in signal based,time series. One assumption is that 
    the causal hierarchical structure of the convolutions may allow the algorithm to create
    functional mappings from different time-stamps that are separete frome ach other. This is
    equivalent to sentences that are apart from each other.
    
    tf.keras.model.Model
    Inherits from Model_Abstract
    
    Future work: gradient based regularization

    """
    
    def __init__(self, input_shape, n_classes, vocab_size, embedding_dim, **kwargs):
        
        """
         Args:
            input_shape (int): shape of data inpit
            n_classes (int): number of classes in the dataset
            vocab_size (int): length of tokenized vocabulary 
            embedding_dim (int): e,bedding mapping vector
        """
        super(NNC_3,self).__init__(input_shape, n_classes, vocab_size, embedding_dim, **kwargs)
        
        self.embedding_layer = Embedding(vocab_size, self.embed_dim, input_length=input_shape)

        self.conv1 = Conv1D(128, 
                            kernel_size=3, 
                            strides=1, 
                            padding='causal',
                            dilation_rate=1,
                            use_bias=True,
                            kernel_initializer='glorot_uniform')
        self.att1= AttentionWeighter(128)
        
        self.conv2 = Conv1D(256, 
                            kernel_size=3, 
                            strides=1, 
                            padding='causal',
                            dilation_rate=2,
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros')
        self.att2 = AttentionWeighter(256)

        self.conv3 = Conv1D(512, 
                            kernel_size=3, 
                            strides=1, 
                            padding='causal',
                            dilation_rate=4,
                            use_bias=True,
                            kernel_initializer='glorot_uniform')
        self.att3 = AttentionWeighter(n_filters=512)
        
                # Convolution + Attention h_(2)
        self.conv4 = Conv1D(1024, 
                            kernel_size=3, 
                            strides=1, 
                            padding='causal',
                            dilation_rate=4,
                            use_bias=True,
                            kernel_initializer='glorot_uniform')
        self.att4 = AttentionWeighter(1024)
        
        self.final_attention = AttentionWeighter(1920)
        
        # Classification stage
        self.linear = Dense(1024, activation='relu')
        self.classifier = Dense(self.num_classes, activation='sigmoid')
        
        # Autobuild
        self.build()
        
    def call(self, x, training=True):
        x = tf.cast(x, dtype=tf.int32)
        x = self.embedding_layer(x) 
        # each convolutional layer is passed into an attentional layer
        hidden1 = self.att1(self.conv1(x))
        hidden2 = self.att2(self.conv2(x))
        hidden3 = self.att3(self.conv3(x))
        hidden4 = self.att4(self.conv4(x))
        
        # concatenate each of the hidden layers and map them into an attentional layer to create 
        # combined mappings
        x = Flatten()(self.final_attention(Concatenate()([hidden1, hidden2, hidden3, hidden4])))
        # Classifier layer
        x = self.classifier(self.linear(x))
        return x
    

class TestBuildModel(unittest.TestCase):
    input_shape = (512)
    error_message = None
    successful_build = False

    def test_input_shape(self):
        self.assertTrue(len(self.input_shape) ==1, "Invalid input shape")

    def test_build_model(self):
        self.model = NNC_3(self.input_shape[0], n_classes=5, vocab_size=30522,embedding_dim=120)
        try:
            self.model.compile(optimizer="adam", loss=CategoricalCrossentropy(), run_eagerly=True)
            self.model.summary()
            self.successful_build = True
        except Exception as message:
            self.successful_build = False
            self.error_message = message
        self.assertTrue(self.successful_build, self.error_message)


if __name__ == '__main__':

    print(f"Tensorflow Version: {tf.__version__}")
    print("Specifiy input dimensions delimited by comma (e.g. (512))")
    TestBuildModel.input_shape = list(map(int, input().strip().split(",")))
    unittest.main()


    

