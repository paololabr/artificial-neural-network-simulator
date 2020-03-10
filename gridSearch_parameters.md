
Parameters for the grid search
===================================

```
params=[
        {'hidden_layer_sizes': [(50,) ,(100,) ,(50,50)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['relu', 'tanh', 'logistic'] }     
    ]
```

Blocks
=========================

```

#Block 1
print ("Block 1")
params=[
        {'hidden_layer_sizes': [(50,)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['relu'] }     
    ]

#Block 2
print ("Block 2")
params=[
        {'hidden_layer_sizes': [(100,)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['relu'] }     
    ]

#Block 3
print ("Block 3")
params=[
        {'hidden_layer_sizes': [(50,50)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['relu'] }     
    ]
	
	
	
	
#Block 4
print ("Block 4")
params=[
        {'hidden_layer_sizes': [(50,)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['tanh'] }     
    ]

#Block 5
print ("Block 5")
params=[
        {'hidden_layer_sizes': [(100,)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['tanh'] }     
    ]

#Block 6
print ("Block 6")
params=[
        {'hidden_layer_sizes': [(50,50)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['tanh'] }     
    ]
	
	
	
	
#Block 7
print ("Block 7")
params=[
        {'hidden_layer_sizes': [(50,)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['logistic'] }     
    ]

#Block 8
print ("Block 8")
params=[
        {'hidden_layer_sizes': [(100,)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['logistic'] }     
    ]

#Block 9
print ("Block 9")
params=[
        {'hidden_layer_sizes': [(50,50)] , 'alpha': [0, 0.001, 0.005, 0.01, 0.05], 'batch_size': [1 , 10 , 50 , 200, 'auto'],
         'learning_rate': ['constant'], 'learning_rate_init': [0.02, 0.1, 0.2], 'momentum': [0., 0.05, 0.7, 0.9],
        'early_stopping': ['True', 'False'], 'activation': ['logistic'] }     
    ]
```

Progress
==========================

| Block         | Who       | Status      | time elapsed | Best MEE           | Notes                     |
|---------------|-----------|-------------|--------------|--------------------|---------------------------|
| block 1       |           | TO DO       |              |                    |                           |
| block 2       |           | TO DO       |              |                    |                           |
| block 3       |           | TO DO       |              |                    |                           |
| block 4       |           | TO DO       |              |                    |                           |
| block 5       |           | TO DO       |              |                    |                           |
| block 6       |           | TO DO       |              |                    |                           |
| block 7       |           | TO DO       |              |                    |                           |
| block 8       |           | TO DO       |              |                    |                           |
| block 9       |           | TO DO       |              |                    |                           |


















