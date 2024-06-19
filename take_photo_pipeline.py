import json
import random
from merge import *
from ground_visualize import *
import re
from snap import *

# Load the JSON data
with open('/home/wa285/rds/hpc-work/Thesis/inference_analysis/misclassified/obj_only.json') as file:
    all_data = json.load(file)

# Randomly sample 100 items if there are enough items
if len(all_data) > 200:
    data = random.sample(all_data, 200)
else:
    data = all_data  # If less than 100, take all

# For storing ply files to be merged
ply_files = []

def extract_obj_id_regex(s):
    # Use a regular expression to find three digits
    match = re.search(r'<OBJ(\d{3})>', s)
    if match:
        return match.group(1)  # Return the group of digits

def main():

    for idx,one_pred in enumerate(data):
        # Get the predicted object ID and gt_id
        pred_id = int(extract_obj_id_regex(one_pred['pred']))
        gt_id = int(one_pred['gt_id'])
        scene_id = one_pred['scene_id']

        #visualise the box and scene
        visualize_runner(scene_id,pred_id,gt_id)
        
        gt_path = os.path.join("vis/", f"{scene_id}/gt_{gt_id}.ply")
        pred_path = os.path.join("vis/", f"{scene_id}/pred_{pred_id}.ply")
        scene_path = f"vis/{scene_id}/mesh.ply"
        output_file = os.path.join("merged_ply/", f"{scene_id}_{idx}.ply")

        #merge the scene in another directory
        merge_ply_files(gt_path, pred_path, scene_path, output_file)

        #take photos of the merged scene

        photo_output_dir = f"photos/{scene_id}_{idx}"

        snapshot_runner(output_file, photo_output_dir)
        # Add photo output directory to the data
        one_pred['full_path'] = f"{scene_id}_{idx}"


    with open("sampled_predictions.json", 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    main()

