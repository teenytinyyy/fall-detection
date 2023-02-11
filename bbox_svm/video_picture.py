import cv2
import os

i = 1
#video_name = "Home_01_RGB"
for j in range(1, 2):
    for i in range(24, 25):
        #cam = "video (" + str(i) + ")"
        videoFile = "../dataset/data/video (" + str(i) + ").avi"
        #videoFile = "../dataset/data/8cam_dataset/chute" + str(j) + "/cam" + str(i) + ".avi"
        #output_path = "../dataset/data/RGB_data/data (" + str(j) + "_" + str(i) + ")/"
        output_path = "../dataset/data/video (" + str(i) + ")/"
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
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if c >= start_frame and c <= end_frame:
                if c % timeF == 0:
                    if rval == False:
                        break
                    else:
                        cv2.imwrite(output_path + "%d.jpg" % (c), frame)
            c += 1

        i += 1
