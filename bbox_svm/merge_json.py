import imp
import os
import json
import tqdm


def merge_json(path_results, path_merges, merge_path):
    merge_file = os.path.join(path_merges, merge_path)
    with open(merge_file, "w", encoding = "utf-8") as f0:
        for file in os.listdir(path_results):
            with open(os.path.join(path_results, file), "r", encoding = "utf-8") as f1:
                for line in tqdm.tqdm(f1):
                    line_dict = json.loads(line)
                    js = json.dumps(line_dict, ensure_ascii = False)
                    f0.write(js + '\n')
                f1.close()
        f0.close()


if __name__ == '__main__':
    path_results, path_merges, merge_path = "../dataset/data/8cam_data/COCOformat", "../dataset/data/8cam_data/COCOformat", "data (1_1)"
    if not os.path.exists(path_merges):
        os.mkdir(path_merges)
    merge_json(path_results, path_merges, merge_path)