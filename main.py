# Python-third party packages
import numpy as np

# Python-standard library packages
from argparse import ArgumentParser
import os
import time
import pickle   

# local packages
from utils import load_data, tokenize_dataset, get_splits, create_tf_data_generator
from model_builder import ModelBuilder
from configs import pipeline_configs
from metrics import calc_metrics


def main(args):
    
    # Creating unique path identifier for model directory
    results_path = os.path.join("./results",f'{args.model}_{str(int(time.time()))}')
    # Data Preprocessing stage
    print('Loading dataset')
    dataset_name = args.dataset
    dataset_dict, num_classes, dataset_dict_keys = load_data(dataset_name)
    # Perform tokenization
    dataset_dict, tokenizer, dataset_config = tokenize_dataset(dataset_dict, **pipeline_configs)
    # Add pipeline configurations for model building
    pipeline_configs.update(dataset_config)
    pipeline_configs['num_classes'] = num_classes
    pipeline_configs['results_path'] = results_path    
    
    # Extracts data splits
    x_train, y_train, x_test, y_test, x_val, y_val = get_splits(dataset_dict, num_classes)
    
    # Create dataset tensor generators: shuffling should only be done in the training set
    dataset_config['input_shape'] = x_train.shape[-1]
    tf_train_dataset= create_tf_data_generator(x_train,y_train, batch_size=pipeline_configs['batch_size'], shuffle=True,reshuffle_each_iteration=True)
    tf_test_dataset= create_tf_data_generator(x_test,y_test, batch_size=pipeline_configs['batch_size'], shuffle=False)
    tf_val_dataset= create_tf_data_generator(x_val,y_val, batch_size=pipeline_configs['batch_size'], shuffle=False)
    
    # Create ModelBuilder to select appropriate model
    model_builder = ModelBuilder(args.model, results_path, **pipeline_configs)
    
    # Building model and retrieving callbacks
    model, callbacks_list = model_builder.build_model()
    # Fitting training dataset
    history = model.fit(
            tf_train_dataset,
            validation_data=tf_val_dataset,
            verbose=1,
            callbacks=callbacks_list, 
            epochs=pipeline_configs.get('epochs', 20))
    
    # Performing testing set querying
    preds = model.predict(tf_test_dataset,verbose=1)
    preds = np.array(preds)
    # Compute performance metrics
    f1,f2,mcc,mse = calc_metrics(preds,y_test)
    # Re-initialize data-generator for the testing set
    tf_test_dataset= create_tf_data_generator(x_test,y_test, batch_size=pipeline_configs['batch_size'], shuffle=False)
    # Compute testing loss and testing accuracy
    result = model.evaluate(tf_test_dataset, verbose=1)

    # Accessing metrics results     
    for name, result in zip(model.metrics_names, result):
        if name == 'accuracy':
            acc = result
        elif name == 'loss':
            loss = result
    
    # Gather metrics results
    results = {
            'acc': acc,
            'loss': loss,
            'f1': f1,
            'f2': f2,
            'mcc': mcc,
            'mse': mse
        }
    
    # saving metrics of model
    with open(os.path.join(results_path, 'summary_model.txt'), 'w') as f:
        f.write(f'model_name: {args.model} \n')
        for metric, value in results.items():
            f.write(f'{metric}: {value} \n')
    
    # saving pipeline configs
    with open(os.path.join(results_path, 'pipeline_configs.json'), 'wb')  as fp:
        pickle.dump(pipeline_configs, fp)    
    
    print(f'Model information stored in path: {results_path}')
    print(f'Loss: {loss:1.4f}\tAccuracy: {acc:1.4f}')
    
if __name__ == '__main__':
    
    # Parser arguments to run file
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="health_fact", help='dataset name')
    parser.add_argument('--model', type=str, default='nnc_1', help='name of model')
    parser.add_argument('--save_loc', type=str, default='./results', help='location to store results')
    args = parser.parse_args()
    main(args)
    