import os
from pathlib import Path
from yaml import load, Loader

def load_configs() -> dict:
  cfg_file = Path("cfg/config.yaml")

  assert cfg_file.is_file()

  with cfg_file.open("r") as f:
    yaml_str = f.read()

  data = load(yaml_str, Loader=Loader)
  return data

def set_environment_variables():
  data = load_configs()
  envs = data.get("envs")
  for k, v in envs.items():
    os.environ[k] = v

def get_configs() -> dict:
  data = load_configs()
  return data.get("configs")
