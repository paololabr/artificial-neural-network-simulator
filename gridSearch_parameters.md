
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

| Block         | size | Who        | time taken/ETA        | Best MEE           | Notes                              |
|---------------|------|------------|-----------------------|--------------------|------------------------------------|
| block 1       | 200  | Lucio      | about 8h              | 1.200393472624     |                                    |
| block 2       | 200  | Lucio      | about 8h              | 1.134549152123     |                                    |
| block 3       | 200  | Lucio      | about 12h             | 1.188549810444     |                                    |
| block 4       | 200  | Lucio      | about 12h             | 1.038886188013     |                                    |
| cp            |  10  | Lucio      | about 1h              | 0.966626228324     | very good results in general       |
| block 1       | 500  | Paolo      |                       | 1.116939722256     |                                    |
| block 2       | 200  | Paolo      |                       | 1.172432842863     |                                    |
| block 5       | 200  | Lucio      | about 8h              | 1.122497639548     | `[(50,50), (100,)`                 |
| block 6       | 200  | Lucio      | about 8h              | 1.062826735599     | `(50,50)`, `["relu", "logistic"]`  |
| block 7       | 200  | Lucio      | about 8h              | 1.049382551239     | only big networks                  |
| block 8       | 200  | Lucio      | about 8h              | 1.084448090462     |                                    |



