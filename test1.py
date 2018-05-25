import cv2
import matplotlib.pyplot as plt
import pandas as pd

vid_path='C:\\Users\\ccc\\Downloads\\udacity-capstone-deeptesla-master\\epochs\\epoch10_front.mkv'
cap=cv2.VideoCapture(vid_path)
ret,img=cap.read()
print(img.shape)
plt.imshow(img)
cap.release()

wheel_sig=pd.read_csv('./epochs/epoch01_steering.csv')
wheel_sig.head()
wheel_sig.wheel.hist(bins=50)
