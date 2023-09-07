from ray import tune

param_space = dict(
    n_batch=tune.grid_search([50, 100]),
    lr=tune.grid_search([2e-3, 1e-3, 5e-3]),
    seq_len=tune.grid_search([100, 200]),
)
