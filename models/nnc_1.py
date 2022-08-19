import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy

from .model_abstract import Model_Abstract
import unittest

class NNC_1(Model_Abstract):
    """
    Natural-language processing convolutional-based algorithm baseline 1.
    This is an attempt to create a simple algorithm containing one embedding with a convolutional
    feature extractor. 

    tf.keras.model.Model class
    In heriits Model_Abstract
    
    """
    
    def __init__(self, input_shape, n_classes, vocab_size, embedding_dim, **kwargs):
        """

        Args:
            input_shape (int): shape of data inpit
            n_classes (int): number of classes in the dataset
            vocab_size (int): length of tokenized vocabulary 
            embedding_dim (int): e,bedding mapping vector
        """
        
        super(NNC_1,self).__init__(input_shape, n_classes, vocab_size, embedding_dim, **kwargs)
        
        # Embedding layer
        self.embedding_layer = Embedding(vocab_size, self.embed_dim, input_length=input_shape)
        self.conv_1 = Conv1D(128, 5, activation='relu')
        self.max_pool = GlobalMaxPooling1D()
        self.dense_1 = Dense(24, activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.classifier =Dense(self.num_classes, activation='sigmoid')
        self.build()
        
        
    def call(self, x, training=True):
        
        x = tf.cast(x, dtype=tf.int32) # cast values to tf.int32 in case they were mapped to tf.int64
        x = self.embedding_layer(x) # map input_ids to embedding
        x = self.conv_1(x) # convolved emedded values
        x = self.max_pool(x) # perform max pooling
        
        # Classification layer with dropout
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
        self.model = NNC_1(self.input_shape[0], n_classes=5, vocab_size=30522,embedding_dim=120)
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


    

