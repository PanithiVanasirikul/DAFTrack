import os
import json

CEPDOF_json_path = '/mnt/ssd1/datasets/fisheye_tracking_datasets/CEPDOF/annotations'
WEPDTOF_json_path = '/mnt/ssd1/datasets/fisheye_tracking_datasets/WEPDTOF/annotations'

CEPDOF_json_files = os.listdir(CEPDOF_json_path)
WEPDTOF_json_files = os.listdir(WEPDTOF_json_path)

CEPDOF_json_files = [os.path.join(CEPDOF_json_path, e) for e in CEPDOF_json_files]
WEPDTOF_json_files = [os.path.join(WEPDTOF_json_path, e) for e in WEPDTOF_json_files]

all_json_files = CEPDOF_json_files + WEPDTOF_json_files

all_json_files.sort()

for file in all_json_files:
    img_names = list()
    img_name_dict = dict()

    with open(file, 'r') as f:
        dataset = json.load(f)

    for e in dataset['annotations']:
        img_names.append(e['image_id'])
        if e['image_id'] not in img_name_dict:
            img_name_dict[e['image_id']] = {'img_name': e['image_id'], 'bboxes': [e['bbox']], 'scores': [1], 'id': [e['person_id']]} #scores is 1 because these are the gt detections
        else:
            img_name_dict[e['image_id']]['bboxes'].append(e['bbox'])
            img_name_dict[e['image_id']]['scores'].append(1)
            img_name_dict[e['image_id']]['id'].append(e['person_id'])
    
    img_names = list(set(img_names))
    img_names.sort()
    output_json = list()
    for e in img_names:
        output_json.append(img_name_dict[e])
    with open(f"./{os.path.basename(file)}", "w") as f:
        json.dump(output_json, f)
    