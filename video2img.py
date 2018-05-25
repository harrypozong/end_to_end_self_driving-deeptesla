import cv2
import imageio
import pandas as pd
import numpy as np
epoch_ids = [i+1 for i in range(10)]
img_data=[[] for i in range(10)]
label_data=[[] for i in range(10)]

for epoch_id in epoch_ids:
    print('---------- processing video {} ----------'.format(epoch_id))
    csv_path =  'epochs/epoch{:0>2}_steering.csv'.format(epoch_id)
    df = pd.read_csv(csv_path)
    label_data[epoch_id-1] = df['wheel'].values
    mkv_path =  'epochs/epoch{:0>2}_front.mkv'.format(epoch_id)
    vid = imageio.get_reader(mkv_path, 'ffmpeg')
    for i in range(len(df)):
        img = cv2.resize(vid.get_data(i),(224,224))
        img_data[epoch_id-1].append(img)
    print(len(img_data))
    print('end of video {},'.format(epoch_id),
          'img_count:',len(img_data[epoch_id-1]),'img_shape:',img_data[epoch_id-1][0].shape,'labels:',label_data[epoch_id-1].shape)

img_data = np.concatenate(img_data, axis=0)
label_data = np.concatenate(label_data, axis=0)
print('-------------end of all---------------','\nimg_data:',img_data.shape,'label_data:',label_data.shape)


##shuffle
p = np.random.permutation(len(img_data))
img_data,label_data = img_data[p],label_data[p]
##split
index = int(len(img_data)*0.8)

train_image,test_image = img_data[:index], img_data[index:]
train_label,test_label = label_data[:index], label_data[index:]
print(train_image.shape,test_image.shape)
print(train_label.shape,test_label.shape)
