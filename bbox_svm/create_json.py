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


def encodeImageForJson(img_arr):
    img_pil = PIL.Image.fromarray(img_arr, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

def img_data(img_path):
	data = labelme.LabelFile.load_image_file(img_path)
	image_data = base64.b64encode(data).decode('utf-8')
	return image_data


def create_json_file(points, imagePath, imageData, imageHeight, imageWidth, file_name):
    # Python 的 dict 類型資料
    dict = {
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": points,
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": imagePath,
  "imageData": imageData,
  "imageHeight": imageHeight,
  "imageWidth": imageWidth
}

    # 將 Python 資料轉為 JSON 格式，儲存至 output.json 檔案
    # indent 參數指定縮排長度
    with open(file_name, "w") as f:
        json.dump(dict, f, indent = 4)