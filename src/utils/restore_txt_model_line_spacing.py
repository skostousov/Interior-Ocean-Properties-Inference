import os

def find_training_info_files(root_dir):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "training_info.txt":
                matches.append(os.path.join(dirpath, filename))
    return matches

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

# Example usage:
saved_models_dir = "saved_models/cluster"
training_info_files = find_training_info_files(saved_models_dir)
for file_path in training_info_files:
    print(file_path)

training_info_dict = {}
for file_path in training_info_files:
    print(f"Processing {file_path}")
    training_info_dict[file_path] = fetch_info(file_path)
    if "model" in training_info_dict[file_path]:
        print(f"Model found in {file_path}")
        with open(file_path + ".model.txt", "w") as out_f:
            model_str = str(training_info_dict[file_path]["model"]).replace("\\n", "\n")
            out_f.write(model_str)
print(training_info_dict.values())