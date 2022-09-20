import files as files

for num_0 in range(1, 2):
    for num_1 in range(2, 5):
        path = "../dataset/data/FDD_data_picture/data ({}_{})".format(num_0, num_1)
        all_files = files.get_files(path)
        files.batch_rename(all_files, '../dataset/data/FDD_data_picture/data ({}_{})/'.format(num_0, num_1), '../dataset/data/8cam_data/labelme/{}_{}_'.format(num_0, num_1))