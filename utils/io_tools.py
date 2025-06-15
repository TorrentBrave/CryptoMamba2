import yaml
import pickle
import pathlib
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_root(file, num_returns=1):
    """
    - file: <class 'str'>
    - tmp: <class 'pathlib.Path'>
        - .parent 会对应的路径进行一次解析，返回上一级目录
        - .resolve() 会返回绝对路径
        - 如果file传进来的是相对路径, .parent 与 .resolve()就会一个显示相对路径, 一个显示绝对路径

    - exists()的设计思想
        - 使用 os 的 os.stat() 来获取文件的状态信息，如果能获取到信息就认为是存在的
    """
    tmp = pathlib.Path(file)
    for _ in range(num_returns):
        tmp = tmp.parent.resolve()
    return tmp


def load_config_from_yaml(path):
    config_file = pathlib.Path(path)
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
        return d
    else:
        raise ValueError(f'Config file ({path}) does not exist.')
    

def save_yaml(data, path):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def save_pickle(data, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data