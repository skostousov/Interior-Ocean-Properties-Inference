
from utils.config import RAW_CONFIG, RELEVANT_CONFIG, PROJECT_ROOT
from pathlib import Path
import copernicusmarine


cfg = RELEVANT_CONFIG
project_root = PROJECT_ROOT

submode = cfg['submode']
submode_cfg = cfg['data'][submode]

data_dir = Path(project_root) / cfg["data"]["data_dir"]
filename = submode_cfg['output_file']
download_dest = (data_dir / filename).resolve()
download_dest.parent.mkdir(parents=True, exist_ok=True)
data_cfg = cfg['data']
features = cfg['data']['features']

# alt_path = (Path(_REPO_ROOT)/self.config['output_dir'] / download_dest.name).resolve()
# print(alt_path)
print(download_dest)
if not download_dest.is_file():
    data_cfg = cfg['data']
    min_latitude = min(data_cfg['latitude_range'])
    max_latitude = max(data_cfg['latitude_range'])
    min_longitude = min(data_cfg['longitude_range'])
    max_longitude = max(data_cfg['longitude_range'])
    start_date = submode_cfg['start_date'].isoformat() + "T00:00:00"
    end_date = submode_cfg['end_date'].isoformat() + "T00:00:00"
    copernicusmarine.subset(
    dataset_id=data_cfg['dataset_id'],
    dataset_version="202311",
    variables=features,
    minimum_longitude=min_longitude,
    maximum_longitude=max_longitude,
    minimum_latitude=min_latitude,
    maximum_latitude=max_latitude,
    start_datetime=start_date,
    end_datetime=end_date,
    output_directory=data_dir,
    output_filename=filename,
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
    coordinates_selection_method="strict-inside",
    netcdf_compression_level=0,
    disable_progress_bar=False,)