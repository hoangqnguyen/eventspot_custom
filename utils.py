import time
import os
import re
import json
import pickle
import gzip
import importlib.util
from pathlib import Path
from typing import Optional, Type
from easydict import EasyDict
from pytorch_lightning.callbacks import Callback


class DoNothing(Callback):
    def on_train_start(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        pass

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, "rt", encoding="ascii") as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 2
        kwargs["sort_keys"] = True
    with open(fpath, "w") as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, "wt", encoding="ascii") as fp:
        json.dump(obj, fp)


def load_pickle(fpath):
    with open(fpath, "rb") as fp:
        return pickle.load(fp)


def store_pickle(fpath, obj):
    with open(fpath, "wb") as fp:
        pickle.dump(obj, fp)


def load_text(fpath):
    lines = []
    with open(fpath, "r") as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def store_text(fpath, s):
    with open(fpath, "w") as fp:
        fp.write(s)


def clear_files(dir_name, re_str, exclude=[]):
    for file_name in os.listdir(dir_name):
        if re.match(re_str, file_name):
            if file_name not in exclude:
                file_path = os.path.join(dir_name, file_name)
                os.remove(file_path)


class Timer:
    def __init__(self, message: str = "Elapsed time"):
        self.message = message

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ) -> None:
        elapsed_time = time.perf_counter() - self.start_time
        print(f"{self.message}: {elapsed_time:.4f} seconds")


def load_config(config_path):
    """
    Dynamically loads a configuration file (Python, JSON, or YAML) and returns the config dictionary.

    Args:
        config_path (str): Path to the config file (.py, .json, or .yaml/.yml).

    Returns:
        EasyDict: Configuration dictionary.
    """
    file_extension = os.path.splitext(config_path)[-1].lower()

    if file_extension == ".py":
        # Load Python config file
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return EasyDict(config_module.config)

    elif file_extension == ".json":
        # Load JSON config file
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return EasyDict(config_dict)

    elif file_extension in [".yaml", ".yml"]:
        # Load YAML config file
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return EasyDict(config_dict)

    else:
        raise ValueError("Unsupported config file format. Use .py, .json, or .yaml/.yml")



def update_config(config, overrides):
    """
    Updates the configuration dictionary with overrides.

    Args:
        config (EasyDict): Original configuration.
        overrides (dict): Overrides to apply.

    Returns:
        EasyDict: Updated configuration.
    """
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Overriding config parameter: {key} = {value}")
        else:
            print(f"Warning: Config has no attribute named '{key}'. Skipping override.")
    return config


def get_num_classes(classes_file):
    """
    Reads the classes file and returns the number of classes.

    Args:
        classes_file (str): Path to the classes file.

    Returns:
        tuple: (num_classes, class_names)
    """
    with open(classes_file, "r") as f:
        class_names = f.read().splitlines()
    # Include 'background' as the first class
    class_names = ["background"] + class_names
    num_classes = len(class_names)
    return num_classes, class_names


def create_experiment_dir(config, run_name=None):
    """
    Creates a unique experiment directory to store checkpoints and config.

    Args:
        config (EasyDict): Configuration dictionary.
        run_name (str, optional): Custom run name. If None, use timestamp.

    Returns:
        str: Path to the experiment directory.
    """
    exp_dir = Path("exp")
    exp_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        from datetime import datetime

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = exp_dir / run_name
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Update config checkpoint_dir to point to the new checkpoints directory
    config.checkpoint_dir = str(checkpoints_dir)

    # Save the config file into the experiment directory for reference

    # Save the config as a JSON file into the experiment directory for reference
    config_save_path = experiment_dir / "config.json"
    # Convert EasyDict to regular dict
    store_json(config_save_path, config, pretty=True)

    return str(experiment_dir)
