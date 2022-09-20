import json
JSON_LOC="../dataset/data/8cam_data/train_mask_rcnn_1_coco/annotations/instances_val2017.json"

#Open JSON
val_json = open(JSON_LOC, "r")
json_object = json.load(val_json)
val_json.close()

for i, instance in enumerate(json_object["annotations"]):
    if len(instance["segmentation"][0]) == 4:
        print("instance number", i, "raises arror:", instance["segmentation"][0])