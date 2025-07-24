import os

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

paths = get_all_txt_contents(root/'lower_res_models')
mega_dict = {"spring": [], "summer": [], "autumn": [], "winter": [], "all": []}
for path in paths:
    info = fetch_info(path)
    mega_dict[info['season']].append(info)



for key, value in mega_dict.items():
    best_infos = []
    max_rmse_dict = {'rmse': float('inf')}
    print(f"{key}: {len(value)} entries")
    for entry in value:
        if  len(best_infos) < 10:
            best_infos.append(entry)
            if float(entry['rmse']) > max_rmse_dict['rmse']:
                max_rmse_dict = entry
        elif float(entry['rmse']) < max_rmse_dict['rmse']:
            best_infos.remove(max_rmse_dict)
            best_infos.append(entry)
            max_rmse_dict=entry
            for info in best_infos:
                if float(info['rmse']) > max_rmse_dict['rmse']:
                    max_rmse_dict = info
    for entry in best_infos:
        print(f"{entry['loss_fn']}, {entry['season']}, {entry['lr']}, {entry['batch_size']}, {entry['rmse']}, {entry['model_specific_args']}, {getattr(entry, 'coarsen', 1)}")

            
            

            


# Example usage:
# folder_path = '/path/to/your/folder'
# all_txts = get_all_txt_contents(folder_path)
# print(all_txts)