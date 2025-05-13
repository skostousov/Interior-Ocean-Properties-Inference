from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).parents[2]

with open(_REPO_ROOT/"configs"/"config.yaml", "r") as f:
    RAW_CONFIG = yaml.safe_load(f)

DATA_DIR = (_REPO_ROOT / RAW_CONFIG["data_dir_relative_to_project_root"]).resolve()