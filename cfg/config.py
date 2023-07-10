import os
import yaml
from pathlib import Path


def load_configs() -> dict:
  cfg_file = Path("config.yaml")
  if not cfg_file.is_file():
    with cfg_file.open("w") as f:
      yaml.dump({"configs": DEFAULT_CONFIG, "envs": DEFAULT_ENVS}, f)

  with cfg_file.open("r") as f:
    yaml_str = f.read()

  return yaml.load(yaml_str, Loader=yaml.Loader)

def set_environment_variables():
  envs = CONFIG.get("envs")
  for k, v in envs.items():
    os.environ[k] = v

def get_configs() -> dict:
  return CONFIG.get("configs")


DEFAULT_CONFIG = {
  'theme': 'fusion',
  'providers': [
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider',
    'DmlExecutionProvider',
    'CPUExecutionProvider'
  ]
}

DEFAULT_ENVS = {
  'QT_MEDIA_BACKEND': 'ffmpeg'
}

CONFIG = load_configs()
