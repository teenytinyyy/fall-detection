import files as files

path = "../dataset/data/8cam_data/COCOformat"
all_files = files.get_files(path)
files.batch_rename(all_files, '../dataset/data/8cam_data/COCOformat/', '../dataset/data/8cam_data/labelme/1_1_')