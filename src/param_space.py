from ray import tune

param_space = dict(
    n_batch=tune.grid_search([10, 50, 100]),
    lr=tune.grid_search([1e-3]),
    seq_len=tune.grid_search([50, 100, 200]),
)
