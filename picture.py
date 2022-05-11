import cv2
import numpy as np


#t1 = "C:\\Users\\Ning\\Desktop\\CV\\frame\\frame217"
#t2 = "C:\\Users\\Ning\\Desktop\\CV\\frame\\frame219"
t1 = cv2.imread('./frame/cam5/frame216.jpg')
t2 = cv2.imread('./frame/cam5/frame217.jpg')
a = cv2.absdiff(t1, t2)
cv2.imwrite("./frame/cam5/new.jpg", a)

t3 = cv2.imread('./frame/cam5/frame217.jpg')
t4 = cv2.imread('./frame/cam5/frame218.jpg')
a1 = cv2.absdiff(t3, t4)
cv2.imwrite("./frame/cam5/new1.jpg", a1)

t5 = cv2.imread('./frame/cam5/frame218.jpg')
t6 = cv2.imread('./frame/cam5/frame219.jpg')
a2 = cv2.absdiff(t5, t6)
cv2.imwrite("./frame/cam5/new2.jpg", a2)

t7 = cv2.imread('./frame/cam5/frame219.jpg')
t8 = cv2.imread('./frame/cam5/frame220.jpg')
a3 = cv2.absdiff(t7, t8)
cv2.imwrite("./frame/cam5/new3.jpg", a3)

t9 = cv2.imread('./frame/cam5/frame220.jpg')
t10 = cv2.imread('./frame/cam5/frame221.jpg')
a4 = cv2.absdiff(t9, t10)
cv2.imwrite("./frame/cam5/new4.jpg", a4)

t11 = cv2.imread('./frame/cam5/frame221.jpg')
t12 = cv2.imread('./frame/cam5/frame222.jpg')
a5 = cv2.absdiff(t11, t12)
cv2.imwrite("./frame/cam5/new5.jpg", a5)

t13 = cv2.imread('./frame/cam5/frame222.jpg')
t14 = cv2.imread('./frame/cam5/frame223.jpg')
a6 = cv2.absdiff(t13, t14)
cv2.imwrite("./frame/cam5/new6.jpg", a6)

t15 = cv2.imread('./frame/cam5/frame223.jpg')
t16 = cv2.imread('./frame/cam5/frame224.jpg')
a7 = cv2.absdiff(t15, t16)
cv2.imwrite("./frame/cam5/new7.jpg", a7)

t17= cv2.imread('./frame/cam5/frame224.jpg')
t18=cv2.imread('./frame/cam5/frame225.jpg')
a8=cv2.absdiff(t17,t18)
cv2.imwrite("./frame/cam5/new8.jpg", a8)

t19 = cv2.imread('./frame/cam5/frame225.jpg')
t20 =cv2.imread('./frame/cam5/frame226.jpg')
a9 = cv2.absdiff(t19, t20)
cv2.imwrite("./frame/cam5/new9.jpg", a9)