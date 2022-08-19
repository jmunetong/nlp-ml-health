import tensorflow as tf
from tensorflow.keras.models import Model

class Model_Abstract(Model):
    """
    Model Subclassed
    This Model has been inherited because future work will focus on modifying training 
    steps within the algorithm. I am thinking of adding a regularization term based
    on the gradients from one pass

    """
    
    def __init__(self, input_shape, n_classes, vocab_size, embedding_dim, **kwargs):
        """
        Args:
            input_shape (int): shape of one observation
            n_classes (int): number of classes in the dataset
            vocab_size (int): length of the vocabulary size for the embedding layer
            embedding_dim (int): embedding output dimension
        """
        super(Model_Abstract,self).__init__(**kwargs)
        self.embed_dim = embedding_dim
        self.in_shape = input_shape
        self.vocab_size = vocab_size
        self.num_classes = n_classes
        
    def train_step(self, data):
            
        # Data unpacking
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value 
            
            # TODO: Add a regularization form here
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def summary(self):
        # Builds graph to identify shape of model
        m = self.build_graph(self.in_shape)
        # Call this instantiated model
        return m.summary()    

    def build_graph(self, input_shape):
        # Method 
        d = tf.keras.Input(input_shape, dtype=tf.int32)
        return Model(inputs=d, outputs=self.call(d, training=True))

    def build(self, **kwargs):
        super().build(input_shape=(None, self.in_shape))



        