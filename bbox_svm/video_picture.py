import cv2
import os

i = 1
#video_name = "Home_01_RGB"
for i in range(1, 9):
    #cam = "video (" + str(i) + ")"
    #videoFile = "./Home_02/Videos/video (" + str(i) + ").avi"
    videoFile = "../dataset/data/8cam_dataset/chute1/cam" + str(i) + ".avi"
    output_path = "../dataset/data/FDD_data_picture/RGBdata_1_" + str(i) + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    vc = cv2.VideoCapture(videoFile)
    start_frame = 0
    end_frame = 2000
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        print('openerror!')
        rval = False

    timeF = 1  # 視頻幀計數間隔次數
    while rval:

        rval, frame = vc.read()

        if rval == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if c >= start_frame and c <= end_frame:
            if c % timeF == 0:
                if rval == False:
                    break
                else:
                    cv2.imwrite(output_path + "%d.jpg" % (c), frame)
        c += 1

    i += 1
