import os
from utils.config import RAW_CONFIG, PROJECT_ROOT, RELEVANT_CONFIG
from pathlib import Path

root = Path(PROJECT_ROOT)

def get_all_txt_contents(root_folder):
    txt_paths = []
    # os.walk recursively traverses all subfolders
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                file_path = os.path.abspath(os.path.join(dirpath, filename))
                txt_paths.append(file_path)
    return txt_paths
    
def fetch_info(info_path):
    with open(info_path, "r") as f:
        info_text = f.read()
        info = {}
        for line in info_text.strip().split('\n'):
            if ':: ' in line:
                key, value = line.split(':: ', 1)
                value = value.strip()
                if value == "None":
                    value = None
                elif value == "True":
                    value = True
                elif value == "False":
                    value = False
                info[key.strip()] = value
        return info

# paths = get_all_txt_contents(root/'lower_res_models')
model_mode = 'dynamic_res_models'
paths = get_all_txt_contents(root/model_mode)


mega_dict = {"spring": [], "summer": [], "autumn": [], "winter": [], "all": []}
for path in paths:
    info = fetch_info(path)
    mega_dict[info['season']].append(info)



for key, value in mega_dict.items():
    max_rmse_dict = {'rmse': float('inf')}
    best_infos = [max_rmse_dict]
    print(f"\n{key}: {len(value)} entries")
    for entry in value:
        if 'rmse' not in entry or (('mld_res' not in entry or float(entry['mld_res'])!=1) and model_mode=='dynamic_res_models'):
            continue
        if  len(best_infos) < 20:
            best_infos.append(entry)
            if float(entry['rmse']) > float(max_rmse_dict['rmse']):
                max_rmse_dict = entry
        elif float(entry['rmse']) < float(max_rmse_dict['rmse']):
            best_infos.remove(max_rmse_dict)
            best_infos.append(entry)
            max_rmse_dict=entry
            for info in best_infos:
                if float(info['rmse']) > float(max_rmse_dict['rmse']):
                    max_rmse_dict = info
    best_infos = sorted(best_infos, key=lambda item: float(item['rmse']))
    for entry in best_infos:
        print(f"{float(entry['rmse']):.2f}, {float(entry.get('mld_res', 000000)):.3f}, {entry.get('loss_fn', 'NA')}, {float(entry.get('lr', 000000)):.5f}, {entry.get('batch_size', 000000)}, {entry.get('model_specific_args', 'Model_sepcific_args_unavailable')}, {entry.get('coarsen', 1)}")

            
            

            


# Example usage:
# folder_path = '/path/to/your/folder'
# all_txts = get_all_txt_contents(folder_path)
# print(all_txts)