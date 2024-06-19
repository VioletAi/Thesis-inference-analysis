import json


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("The file was not found.")
        return None


file_path = 'output/scene_only.json'  

data = read_json_file(file_path)

pred_in_ref_captions = []
pred_not_in_ref_captions = []

if data is not None:

    for entry in data:
        if entry["pred"] in entry["ref_captions"]:
            pred_in_ref_captions.append(entry)
        else:
            pred_not_in_ref_captions.append(entry)

    with open('pred_in_ref_captions.json', 'w') as file:
        json.dump(pred_in_ref_captions, file, indent=4)

    with open('pred_not_in_ref_captions.json', 'w') as file:
        json.dump(pred_not_in_ref_captions, file, indent=4)

    print("Files saved successfully!")
else:
    print("Failed to read data from file.")
