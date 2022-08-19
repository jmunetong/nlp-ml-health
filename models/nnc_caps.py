import tensorflow as tf
from .model_abstract import Model_Abstract
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv1D, Dense, Embedding
from tensorflow.keras.losses import CategoricalCrossentropy

import unittest

from .layers import FlattenCaps, FCCaps, PrimaryCaps

class NNC_Caps(Model_Abstract):
    """
    Capsule neural network with KDE-Dynamic routing implementation
    `https://proceedings.neurips.cc/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html`
    
    Model in herits abstract class Model_Abstract
    """

    def __init__(self, 
                 vocab_size,
                 num_classes,
                 dim_capsule, 
                 num_compressed_capsule, 
                 embedding_dim, 
                 max_length,
                 **kwargs):
        """
        
        Args:
            vocab_size (int): size of vocabulary
            num_classes (int): number of classes in the dataset
            dim_capsule (int): length of capsule of neural network
            num_compressed_capsule (int): reduction of length vector for memory capacity
            embedding_dim (int): length of embedding vector for input_id mapping
            max_length (int): maximum sentence length from input_id

        """
        super(NNC_Caps, self).__init__(**kwargs)
        self.in_shape = vocab_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.num_compressed_capsule = num_compressed_capsule 
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = max_length
        # add the embedding class here
        self.embed = Embedding(vocab_size, embedding_dim, input_length=max_length)

        # N-gram size of words instantiation
        self.ngram_size = [2,4,8]

        # Instantiating the feature extraction convolutions based on different lengths of words
        # We chose 3 lengths, but this can be modified to fit more, if necessary
        self.convs_doc = [Conv1D(32, K, strides=2, kernel_initializer='glorot_uniform') for K in self.ngram_size]
        self.primary_capsules = PrimaryCaps(num_capsules=self.num_classes, 
                                                out_filters=32, 
                                                kernel_size=1, 
                                                stride=1)

        self.flatten_capsules = FlattenCaps() 

        # Initializing weights of model       
        self.W = self.add_weight(
                                    shape=(self.vocab_size, self.num_compressed_capsule), 
                                    initializer="glorot_uniform", 
                                    trainable=True
                                    )
    
        self.fc_capsules_doc_child = FCCaps(output_capsule_num=self.num_classes, 
                                            input_capsule_num=self.num_compressed_capsule,
                            	            in_channels=self.dim_capsule, 
                                            out_channels=self.dim_capsule)
        self.classifier = Dense(self.num_classes, activation="sigmoid")
        self.build()

    def call(self, data):
        
        # Reducing dimensionality of capsules
        def compression(poses, W):
            poses = tf.transpose(tf.matmul(tf.transpose(poses, perm=[0,2,1]), W), perm=[0,2,1])
            activations = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(poses),2))
            return poses, activations
        # compute embedding
        data = self.embed(data)
        nets_list = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](data) # feature extraction based on the number of 
            nets_list.append(nets)
        # concatenate all feature extractions
        nets_doc = tf.concat((nets_list[0], nets_list[1], nets_list[2]), 2)
        # preprocessing of capsules
        poses_doc, activations_doc = self.primary_capsules(nets_doc)
        # computing capsules dynamic routing algorithm
        poses, activations = self.flatten_capsules(poses_doc, activations_doc)
        poses, activations = compression(poses, self.W)
        _, activations = self.fc_capsules_doc_child(poses, activations)
        # classification stage
        activations = self.classifier(activations)
        return activations

    

class TestBuildModel(unittest.TestCase):
    input_shape = (512)
    error_message = None
    successful_build = False

    # def test_input_shape(self):
    #     self.assertTrue(len(self.input_shape) == 3, "Invalid input shape")

    def test_build_nnc_caps(self):
        self.model = NNC_Caps(self.input_shape, num_classes=12, dim_capsule=20, num_compressed_capsule=10)
        self.model.compile(optimizer="adam", loss=CategoricalCrossentropy, run_eagerly=True)
        self.model.summary()
        self.successful_build = True

        try:
            self.model.compile(optimizer="adam", loss=CategoricalCrossentropy(), run_eagerly=True)
            self.model.summary()
            self.successful_build = True
            
        except Exception as message:
            self.successful_build = False
            self.error_message = message
        self.assertTrue(self.successful_build, self.error_message)
        self.assertTrue(
            self.out_shape == self.input_shape, "Invalid decoder output shape"
        )
        
if __name__ == "__main__":

    print(f"Tensorflow Version: {tf.__version__}")
    print("Specifiy input dimensions delimited by comma (e.g. 100,100,3)")
    TestBuildModel.input_shape = int(input())
    unittest.main()
