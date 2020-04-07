
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
         'activation': ['relu', 'tanh', 'logistic'],
         'weights_init_fun': ["random_uniform", "random_normal"], 
         "weights_init_value": [0.2, 0.8]
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
| block 5       | 200  | Lucio      | about 3h              | 1.080942114956097      |  n_iters_no_change = 20            |
| block 6       | 200  | Lucio      | about 3h              | 1.134746906081432      |  n_iters_no_change = 20            |
| block 7       | 200  | Lucio      | about 3h              | 1.060477383000672      |  n_iters_no_change = 20            |
| block 8       | 200  | Lucio      | about 3h              | 1.065416728357821      |  n_iters_no_change = 20            |
| block 9       | 600  | Lucio      |                       | 1.088186529922876      |  weights_init_fun="random_normal"  |
| block 10      | 600  | Lucio      |                       | 1.102934158063803      |  weights_init_fun="random_normal"  |
| block 11      | 600  | Lucio      |                       | 1.047133861537207      |  weights_init_fun="random_normal"  |
| block 12      | 600  | Lucio      |                       | 1.114407996994462      |  weights_init_fun="random_normal"  |
| block 1       |      | Paolo      |                       | 1.082011480279572      |                                    |
| block 2       |      | Paolo      |                       | 1.053694407847668      |                                    |
| block 3       |      | Paolo      |                       | 1.057056711318689      |                                    |
| block 4       |      | Paolo      |                       | 1.022297435896925      |                                    |
| block 5       |      | Paolo      |                       | 1.101321785512985      |                                    |
| block 7       |      | Paolo      |                       | 1.036575560431177      |                                    |
| block 8       |      | Paolo      |                       | 1.134017362094504      |                                    |
| block 9       |      | Paolo      |                       | 1.078560097658925      | weights_init_fun="random_normal"   |
| block 10      |      | Paolo      |                       | 0.993331715990709      | weights_init_fun="random_normal"   |

