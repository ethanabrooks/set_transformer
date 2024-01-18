from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def check_alphabetical_order(d: DictConfig, name: str):
    out_of_order_keys = is_alphabetical_order(OmegaConf.to_container(d))
    if out_of_order_keys:
        print(f"The following keys are not in alphabetical order in {name}:")
        for keys in out_of_order_keys:
            print(".".join(keys))
        exit(1)


def get_config(config_name: str):
    root = Path("configs")
    config_path = root / f"{config_name}.yml"
    config = OmegaConf.load(config_path)
    check_alphabetical_order(config, str(config_path))

    def get_parents(config: DictConfig):
        while True:
            inherit_from = config.pop("inherit_from", None)
            if inherit_from is None:
                break
            base_config_path = config_path.parent / Path(inherit_from).with_suffix(
                ".yml"
            )
            config = OmegaConf.load(base_config_path)
            check_alphabetical_order(config, str(base_config_path))
            yield config

    parents = reversed(list(get_parents(config)))
    merged = OmegaConf.merge(*parents, config)
    if not merged["train_trajectories"]:
        for key in ("train_data_args", "test_data_args"):
            merged[key] = OmegaConf.merge(merged["data_args"], merged[key])
        del merged["data_args"]
    resolved = OmegaConf.to_container(merged, resolve=True)
    return resolved


def is_alphabetical_order(d, parent_keys=None):
    """Recursively check if all keys in the dictionary are in alphabetical order
    and return the keys that are out of order.
    """
    if parent_keys is None:
        parent_keys = []
    if not isinstance(d, dict):
        return []
    keys = list(d.keys())
    out_of_order_keys = [
        parent_keys + [k1] for k1, k2 in zip(keys, sorted(keys)) if k1 != k2
    ]
    for k, v in d.items():
        out_of_order_keys.extend(is_alphabetical_order(v, parent_keys + [k]))
    return out_of_order_keys
