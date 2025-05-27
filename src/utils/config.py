from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).parents[2]

cfg_path = _REPO_ROOT / "configs" / "config.yaml"

with open(cfg_path, "r") as f:
    RAW_CONFIG = yaml.safe_load(f)

RAW_CONFIG["project_root"] = str(_REPO_ROOT)

with cfg_path.open("w") as f:
    yaml.safe_dump(RAW_CONFIG, f, sort_keys=False)

DATA_DIR = (_REPO_ROOT / RAW_CONFIG["data_dir_relative_to_project_root"]).resolve()
SAVED_MODELS_DIR = (_REPO_ROOT / RAW_CONFIG["model_save_dest"]).resolve()
FEATURES = RAW_CONFIG["features"]
EARLY_STOP = RAW_CONFIG['early_stopping_thresh']
MONTHLY_CONFIG = RAW_CONFIG["monthly mode"]

def fetch_datasets():
    pass

if __name__=="__main__":
    print(FEATURES)
    print(RAW_CONFIG["monthly mode"])