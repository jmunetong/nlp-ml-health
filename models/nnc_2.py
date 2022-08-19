import tensorflow as tf
from .model_abstract import Model_Abstract
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.losses import CategoricalCrossentropy

import unittest

class NNC_2(Model_Abstract):
    """
    Second model is just an expanded version of the first model with additional convolved and max-pooled layers
    
    tf.keras.model.Models 
    Inherits from Model_Abstract
    This model will be modified to add regularization based on gradients and hessian matrix

    """
    
    def __init__(self, input_shape, n_classes, vocab_size, embedding_dim, **kwargs):
        """
         Args:
            input_shape (int): shape of data inpit
            n_classes (int): number of classes in the dataset
            vocab_size (int): length of tokenized vocabulary 
            embedding_dim (int): e,bedding mapping vector
        """
        super(NNC_2,self).__init__(input_shape, n_classes, vocab_size, embedding_dim, **kwargs)

        self.embedding_layer = Embedding(vocab_size, self.embed_dim, input_length=input_shape) # embedding mapping
        
        # Convolution block
        self.conv_1 = Conv1D(128, 2, activation='relu')
        self.conv_2 = Conv1D(256, 2, activation='relu')
        self.max_pool_2 = MaxPooling1D(pool_size=6)
        self.conv_3 = Conv1D(512, 2, activation='relu')
        self.max_pool_3 = MaxPooling1D(pool_size=6)
        self.conv_4 = Conv1D(1024, 2, activation='relu')
        self.max_pool_4 = MaxPooling1D(pool_size=2)
        
        # Global Max poolign and classification layers
        self.maxpool = GlobalMaxPooling1D()
        self.dense_1 = Dense(2048, activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.classifier =Dense(self.num_classes, activation='sigmoid')
        self.build()
        
    def call(self, x, training=True):
        x = tf.cast(x, dtype=tf.int32)
        x = self.embedding_layer(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.max_pool_3(x)
        x = self.conv_4(x)
        x = self.max_pool_4(x)
        x = self.maxpool(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
class TestBuildModel(unittest.TestCase):
    input_shape = (512)
    error_message = None
    successful_build = False

    def test_input_shape(self):
        self.assertTrue(len(self.input_shape) ==1, "Invalid input shape")

    def test_build_model(self):
        self.model = NNC_2(self.input_shape[0], n_classes=5, vocab_size=30522,embedding_dim=120)
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


    

