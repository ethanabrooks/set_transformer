from ray import tune

param_space = dict(
    seed=tune.grid_search([0, 1]),
)
