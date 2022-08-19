from contextlib import redirect_stdout
from tensorflow.keras.metrics import CategoricalAccuracy as accuracy
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as callbacks
from os.path import join
from os import makedirs

class ModelBuilder(object):
    """
    Class object that instantiate the necessary parameters to build and compile
    one of the neural network candidates
    """
    def __init__(self,model_name, path,**kwargs):
        """
        __init__ method

        Args:
            model_name (str): keyed name of model
            path (str): relative path to directory where results will be stored
        """
        ## Access Keyword arguments
        
        self.model_name = model_name  # model name
        self.input_shape = kwargs.get('input_shape', (512)) # shape of an observation
        self.n_classes = kwargs.get('num_classes', 5) # number of classes in the dataset
        self.metric = kwargs.get('metric', accuracy(name='accuracy')) # metrics to watch
        self.results_path = path # path to store model's weights
        self.training_loss = kwargs.get('loss','categorical_crossentropy') # loss
        self.watch = kwargs.get('monitor','val_loss') # watcher
        self.wait = kwargs.get('patience',10) # patience instantiator
        self.verbose = kwargs.get('verbose',False) # logs output
        self.embedding_dim = kwargs.get('embedding_dim',512) #embedding layer dimension
        self.opt = Adam(learning_rate=3e-4,amsgrad=True) # Learning rate
        self.run_eagerly = True # must run eagerly for sub-classed models
        self.weights_path = join(self.results_path,'weights','weights.h5') # path to store weights
        self.params_args = kwargs # remaining parameters
        self.vocab_size = kwargs.get('vocab_size', self.input_shape) # vocabulary size for embedding layer
        
        makedirs(join(self.results_path,'weights'),exist_ok=True)

    def build_model(self):
        
        # Import appropriate model
        if self.model_name == 'nnc_1': 
            from models import NNC_1 as mod
        elif self.model_name == 'nnc_2':
            from models import NNC_2 as mod
        elif self.model_name == 'nnc_3':
            from models import NNC_3 as mod
        elif self.model_name == 'nnc_caps':
            from models import NNC_Caps as mod
        else: raise ValueError(f'{self.model_name} is not a valid model choice!')
       
       # Instantiate model
        model = mod(
            input_shape=self.input_shape,
            n_classes=self.n_classes,
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size
            )
         
        model.compile(self.opt,self.training_loss,metrics=self.metric, run_eagerly=self.run_eagerly)
        cblist = self.__setup_callbacks()
        with open(f'{self.results_path}/architecture_summary.txt','w') as f:
            with redirect_stdout(f): model.summary()
        if self.verbose: model.summary()
        return model,cblist

    def __setup_callbacks(self):
        
        # Callback to automate early stopping based on number of epochs and an epsilon rate value change in validation loss
        early_stop = callbacks.EarlyStopping(
            monitor=self.watch, 
            patience=self.wait, 
            min_delta=0.001,
            verbose=1)
        # Callback to store best weights
        checkpoint = callbacks.ModelCheckpoint(
            filepath=self.weights_path, 
            monitor=self.watch, 
            verbose=0,
            save_best_only=True,
            save_weights_only=True)
        # Callback to modify learning-rate during training
        reduce_on_plateau = callbacks.ReduceLROnPlateau(
            monitor=self.watch,
            factor=0.5,
            patience=self.wait//2,
            verbose=0,
            mode='min',
            min_delta=0.001,
            min_lr=1e-8)
        return [early_stop,checkpoint,reduce_on_plateau]

