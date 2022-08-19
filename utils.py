from datasets import load_dataset, load_dataset_builder, get_dataset_config_names
import tensorflow as tf
from transformers import  AutoTokenizer
import numpy as np

def load_data(dataset_name, show_description=False, label_key='label'):
    
    """
    Function to load the dataset-dict from HugginFace reposutory direcotry
    
    Args:
        dataset_name (str): denoting the name of dataset stored in cloud-repository
        show_description (bool): to denote whether to print feature and datadescription
        label_key (str): string denoting the key in which the label values are located within the dictionary

    Returns:
        DatasetDict: data structure containing training, testing and/or validation keys
        int: referring to the number of classes in the dataset
        
    """
    
    # Import dataset. We are not specifying the datasplit yet. We first want to explore the splits available in the dataset
    dataset_dict = load_dataset(dataset_name)
    
    # Whether to print an overall description of the dataset
    if show_description:
        ds_builder = load_dataset_builder(dataset_name)
        ds_builder.info.description
        print(ds_builder.info.description)
        print(ds_builder.info.features)
    
    dataset_dict_keys = list(dataset_dict.keys())
    # (2) Compute the number of classes in the dataset
    classes = len(np.unique(dataset_dict[dataset_dict_keys[0]][label_key]))
    

    return dataset_dict, classes, dataset_dict_keys

def tokenize_dataset(dataset_dict, 
                      tokenizer="bert-base-uncased",
                      do_lower_case=True,
                      chosen_columns =['claim', 'main_text'],
                      truncation=True, 
                      padding='max_length',
                      remove_columns =[],
                      batched=True,
                      **kwargs):

    """
    Function to select columns for data parsing and tokanization. This function also loads
    a pretrained tokenizer. Columns that are not in chosen_columns argument will be
    removed from DatasetDict
    Args:
        dataset_dict (DatasetDict): datastructure containing the dataset
        tokenizer (str): name of tokenizer stored in Transformers python package
        do_lower_case (bool): whether to lower case input
        chosen_columns (list(str)): list of selected columns to perform tokenization
        truncation (bool): whether to perform truncation
        padding (str): type of padding, default is 'max_length'
        remove_columns (list(str)): name of columns to remove
        batched (bool): whether to perform batching in tokenization
    

    Returns:
        DatasetDict: the same dataset data-strcuture containing the data now tokenized
        AutoTokenizer: tokenizer model used for parsing-tokenizing mappings
        Dict: dictionary containing metadata from the data-tokenization steps
    """

    # load tokenizer, we are choosing a pretrained model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=do_lower_case)
    
    # selecting the columns to perform tokenization. In this case, we used the 'claim' and 'main_text  columns 
    if len(chosen_columns) ==2:
        dataset_dict = dataset_dict.map(lambda observation:  tokenizer(observation[chosen_columns[0]], observation[chosen_columns[1]],  
                                                                    truncation=truncation, 
                                                                    padding=padding), 
                                        batched=batched)
    else:
        assert len(chosen_columns) == 1,  f"The number of chosen columns must be 1 or 2"
        dataset_dict = dataset_dict.map(lambda observation:  tokenizer(observation[chosen_columns[0]],  
                                                                    truncation=truncation, 
                                                                    padding=padding), 
                                        batched=batched)

    # Remap labels
    dataset_dict = dataset_dict.map(lambda examples: {"labels": examples["label"]}, batched=batched)
    
    # Add meta-data from tokenizer
    dataset_config = { 'vocab_size':tokenizer.vocab_size,
                        'max_length': tokenizer.model_max_length,
                        'columns_removed': remove_columns}
    
    # Whether to remove columns
    if remove_columns is not None or len(remove_columns) > 0:
        dataset_dict = dataset_dict.remove_columns(remove_columns)
        
    return dataset_dict, tokenizer, dataset_config

def get_splits(dataset_dict, n_classes):
    """
    Function that extracts dataset splits and one-hot encodes labels for each of the splits

    Args:
        dataset_dict, (DatasetDict): dictionary-based datastructure that contains the trianing,testing
            and validation sets
        n_classes (int): number of classes in the dataset

    Returns:
        np.ndarray: x-training observations
        np.ndarray: x-training labels (one-hot-encoded)
        np.ndarray: x-testing observations
        np.ndarray: y-testing labels (one-hot-encoded)
        np.ndarray: x-validation observations
        np.ndarray: y-validation labels (one-hot-encoded)

    """
    x_train = np.array(dataset_dict['train']['input_ids']).astype(np.int32)
    y_train = np.eye(n_classes)[dataset_dict['train']['labels']].astype(np.float32)
    x_test = np.array(dataset_dict['test']['input_ids']).astype(np.int32)
    y_test = np.eye(n_classes)[dataset_dict['test']['labels']].astype(np.float32)
    x_val = np.array(dataset_dict['validation']['input_ids']).astype(np.int32)
    y_val = np.eye(n_classes)[dataset_dict['validation']['labels']].astype(np.float32)
    return x_train, y_train, x_test, y_test, x_val, y_val

def create_tf_data_generator(x,y, batch_size=20, shuffle=False,reshuffle_each_iteration=True):
    """
    Function to instantiate tf.data.Dataset generators 

    Args:
        x (np.ndarray): x observations
        y (np.ndarray): y labels (one-hot-encoded)
        batch_size (int, optional): _description_. Defaults to 20.
        shuffle (bool, optional): _description_. Defaults to False.
        reshuffle_each_iteration (bool, optional): _description_. Defaults to True.

    Returns:
        tf.data.Dataset: data-generator
    """
    
    tf_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        tf_dataset = tf_dataset.shuffle(len(x), reshuffle_each_iteration=reshuffle_each_iteration) # Shuffle training set
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset
