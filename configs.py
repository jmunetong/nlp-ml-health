

# Python file containing the dictionary configurations for running the pipeline
# this has been done to compress all possible paramters in one file
pipeline_configs = {
    'tokenizer': "bert-base-uncased",
    'do_lower_case': True,
    'chosen_columns': ['claim', 'main_text'],
    'truncation':True, 
    'padding':'max_length',
    'remove_columns' :['claim_id', 'claim', 'date_published', 
                       'explanation', 'fact_checkers', 
                       'main_text', 'sources','subjects', 
                       'label'],
    'batched':True,
    'embedding_dim': 120,
    'loss':'categorical_crossentropy',
    'monitor':'val_loss',
    'patience': 10,
    'verbose':False,
    'batch_size': 20,
    'epochs':10
}