import json
import codecs
import base64
import io
import numpy as np
import PIL
import labelme
from networkx.generators.tests.test_small import null


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def encode_image_for_json(img_arr):
    img_pil = PIL.Image.fromarray(img_arr, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    enc_data = codecs.encode(data, 'base64').decode()
    enc_data = enc_data.replace('\n', '')
    return enc_data


def img_data(img_path):
    data = labelme.LabelFile.load_image_file(img_path)
    image_data = base64.b64encode(data).decode('utf-8')
    return image_data


def create_json_file(points, image_path, image_data, image_height, image_width, file_name):

    points = list(points[0])

    points = [[int(x) for x in p] for p in points]

    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "person",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": image_path,
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 將 Python 資料轉為 JSON 格式，儲存至 output.json 檔案
    # indent 參數指定縮排長度
    with open(file_name, "w") as f:
        json.dump(json_data, f, indent=4)
