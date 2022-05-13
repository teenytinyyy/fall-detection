# import cv2
#
#
# def get_images_from_video(video_name, time_F):
#     video_images = []
#     vc = cv2.VideoCapture(video_name)
#     c = 1
#
#     if vc.isOpened():  # 判斷是否開啟影片
#         rval, video_frame = vc.read()
#     else:
#         rval = False
#
#     while rval:  # 擷取視頻至結束
#         rval, video_frame = vc.read()
#
#         if (c % time_F == 0):  # 每隔幾幀進行擷取
#             video_images.append(video_frame)
#         c = c + 1
#     vc.release()
#
#     return video_images
#
#
# time_F = 8  # time_F越小，取樣張數越多
# video_name = 'cam1.avi'  # 影片名稱
# video_images = get_images_from_video(video_name, time_F)  # 讀取影片並轉成圖片
# start_frame = 120
# end_frame = 144
#
# for i in range(0, len(video_images)):  # 顯示出所有擷取之圖片
#     cv2.imshow('windows', video_images[i])
#     cv2.waitKey(0)    #等待與讀取使用者按下的按鍵，而其參數是等待時間（單位為毫秒），若設定為 0 就表示持續等待至使用者按下按鍵為止
#     cv2.imwrite('output.jpg', img[i])
#
# cv2.destroyAllWindows


import cv2

videoFile = "C:\\Users\\Ning\\Downloads\\dataset\\dataset\\chute16\\cam2.avi"
outputFile = './frame/chute16/cam2/frame'
vc = cv2.VideoCapture(videoFile)
start_frame = 800
end_frame = 2000
c = 1
if vc.isOpened():
    rval, frame = vc.read()
else:
    print('openerror!')
    rval = False

timeF = 5  #視頻幀計數間隔次數
while rval:
    #print(1)
    print(c)
    rval, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if c >= start_frame and c <= end_frame:
        if c % timeF == 0:
            print(2)
            cv2.imwrite(outputFile + str(int(c / timeF)) + '.jpg', frame)
            #img = outputFile + str(int(c / timeF)
    c += 1
    #for c in range (start_frame,end_frame):
    #    cv2.imwrite(cv2.absdiff(c, c+1)) + '.jpg', frame)
    cv2.waitKey(1)
vc.release()

# img = './frame/frame'
#
# t0 = img.read()[1]
# t1 = img.read()[1]
# cv2.imwrite((cv2.absdiff(t0, t1)) + '.jpg', frame)