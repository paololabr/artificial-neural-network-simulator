
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

| Block         | size | Who        | time taken/ETA        | Best MEE           | Notes                     |
|---------------|------|------------|-----------------------|--------------------|---------------------------|
| block 1       | 200  | Lucio      | about 8h              | 1.200393472624     |                           |
| block 2       | 200  | Lucio      | about 8h              | 1.1345491521232    |                           |
| block 3       | 200  | Lucio      | ETA 2020-03-13T14:00  |                    |                           |
| block 4       | 200  | Lucio      | ETA 2020-03-13T14:00  |                    |                           |
| cp            |  10  | Lucio      | ETA 2020-03-13T16:00  |                    |                           |



