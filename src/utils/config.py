from pathlib import Path
import yaml

_REPO_ROOT = Path(__file__).parents[2]

cfg_path = _REPO_ROOT / "configs" / "config.yaml"

with open(cfg_path, "r") as f:
    RAW_CONFIG = yaml.safe_load(f)

RAW_CONFIG["project_root"] = str(_REPO_ROOT)

# with cfg_path.open("w") as f:
#    yaml.safe_dump(RAW_CONFIG, f, sort_keys=False)

PROJECT_ROOT = RAW_CONFIG["project_root"]
FEATURES = RAW_CONFIG["features"]
mode = RAW_CONFIG["mode"]
RELEVANT_CONFIG = RAW_CONFIG[mode]

if __name__=="__main__":
    print(FEATURES)
    print(RELEVANT_CONFIG)