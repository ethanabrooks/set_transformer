from ray import tune

param_space = dict(
    n_batch=tune.grid_search([10, 50, 100]),
    lr=tune.grid_search([1e-5, 1e-4, 1e-3]),
    seq_len=tune.grid_search([50, 100, 200]),
    max_order=tune.grid_search([1, 2]),
)
