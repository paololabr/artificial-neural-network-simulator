
Parameters for the grid search
===================================

```
params=[
        {
          'hidden_layer_sizes': [(10,10), (20,), (50,) ,(100,), (50,50)],
          'alpha': [0., 0.05], 
          'batch_size': [1, 5, 10, 50, 100, 'auto', 500, len(data)],
          'learning_rate': ['constant', 'adaptive', 'linear'],
          'learning_rate_init': [0.001, 0.1],
          'momentum': [0., 0.9],
          'early_stopping': [True, False],
          'activation': ['relu', 'tanh', 'logistic'] 
        }
    ]
```


Progress
==========================

| Block         | size | Who        | time taken/ETA        | Best MEE               | Notes                              |
|---------------|------|------------|-----------------------|------------------------|------------------------------------|
| block 1       | 600  | Lucio      | about 3h              | 1.072956110751453      |                                    |
| block 2       | 600  | Lucio      | about 3h              | 1.082714786984274      |                                    |
| block 3       | 600  | Lucio      | about 3h              | 1.118019330474649      |                                    |
| block 4       | 600  | Lucio      | about 3h              | 1.115904106139977      |                                    |
