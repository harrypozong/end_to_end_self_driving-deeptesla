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
img_data=[]
label_data=[]

def train_data_generator(imgs,wheels,purpose):  
    _x = np.zeros((FLAGS.batch_size, FLAGS.img_w, FLAGS.img_h, FLAGS.img_c), dtype=np.float)
    _y = np.zeros(FLAGS.batch_size, dtype=np.float)
    out_idx = 0 
    while 1:
        frame_idx = np.random.randint(n_purpose)        ## Find angle
        angle = wheels[purpose][frame_idx]        ## Find frame
        img = imgs[purpose][frame_idx]        ## Implement data augmentation
        if img is not None:
            _x[out_idx] = img
            _y[out_idx] = angle
            out_idx += 1
        if out_idx >= 16:
            yield _x, _y            # Reset the values back
            _x = np.zeros((16, 224, 224, 3), dtype=np.float)
            _y = np.zeros(16, dtype=np.float)
            out_idx = 0

base_model=VGG16(weights='imagenet', include_top=False)
#base_model.load_weights('vgg16_weights_th_dim_ordering_th_kernels_notop.h5')

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
index = int(len(img_data)*0.8)



for epoch_id in epoch_ids:
    csv_path =  'epochs/epoch{:0>2}_steering.csv'.format(epoch_id)
    df = pd.read_csv(csv_path)
    label_data=df['wheel'].values
    mkv_path =  'epochs/epoch{:0>2}_front.mkv'.format(epoch_id)
    vid = imageio.get_reader(mkv_path, 'ffmpeg')
    for i in range(len(df)):
        img = cv2.resize(vid.get_data(i),(224,224))
        img_data.append(img)
    train_image,test_image = img_data[:index], img_data[index:]
    train_label,test_label = label_data[:index], label_data[index:]
    model.fit_generator(
        train_data_generator(train_image,train_label,'train'),
        samples_per_epoch=FLAGS.train_batch_per_epoch * 16,
        nb_epoch=10,
        validation_data=val_data_generator(train_image,train_label,'val'),
        nb_val_samples=16,
        verbose=1)
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
