from keras.models import Model
from keras.optimizers import SGD,RMSprop
from keras.callbacks import TensorBoard
from keras.layers import Input, Flatten, Dropout,Dense,Reshape,Convolution2D, MaxPooling2D,GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
import cv2
import imageio
import pandas as pd  #数据分析模块
import numpy as np
import matplotlib.pyplot as plt

epoch_ids = [i+1 for i in range(10)]  #[1,2,3,4,5,6,7,8,9,10]
img_data=[[] for i in range(10)]      #[[],[],[],[],[],[],[],[],[],[]]
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
img_data,wheels = img_data[p],label_data[p]
print(len(img_data))
##split
index = int(len(img_data)*0.8)

train_image,test_image = img_data[:200], img_data[300:316]
train_label,test_label = label_data[:200], label_data[300:316]
print(train_image.shape,test_image.shape)
print(train_label.shape,test_label.shape)

base_model=VGG16(weights='imagenet', include_top=False)
#base_model.load_weights('vgg16_weights_th_dim_ordering_th_kernels_notop.h5')
print("ccc")
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.7)(x)
output = Dense(1)(x)
model = Model(input=base_model.input,output=output)

for layer in base_model.layers:
    layer.trainable = False

opt = RMSprop(lr=1e-6)
model.compile(optimizer='Adadelta',
              loss='mse')

model.fit(train_image,train_label,validation_split=0.2,batch_size=16, callbacks=[TensorBoard(log_dir='./log')])
model.evaluate(test_image, test_label)

model.save('./modelsvgg16/model.h5')
with open('./modelsvgg16/model.json', 'w') as f:
    f.write(model.to_json())

model.evaluate(test_image, test_label)
print("evaluate over")
pred = model.predict(train_image[0:20])
plt.plot(pred,'r')
plt.plot(train_label[0:20],'b')
plt.show()
