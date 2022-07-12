#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
import cv2

from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imutils


# In[5]:


store_image=[]
train_path='C:/Users/80965/train'
fps=5
train_videos=os.listdir('train_path')
train_images_path='C:/Users/80965/train/frames'
os.makedirs(train_images_path)


# In[6]:


print(train_videos)


# In[7]:


def store_inarray(image_path):
    image=load_img(image_path)
    image=img_to_array(image)
    image=cv2.resize(image, (227,227), interpolation = cv2.INTER_AREA)
    gray=0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2]
    store_image.append(gray)


# In[8]:


for video in train_videos:
    os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(train_path,video,fps,train_path))
    #print(video)
    vidcap = cv2.VideoCapture("./train_path/"+video)
    success,image = vidcap.read()
    count = 1
    #print(success,image)
    #print(vidcap)
    while success:
        cv2.imwrite("C:/Users/80965/train/frames/%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1
    images=os.listdir(train_images_path)
    #print(images)
    for image in images:
        image_path=train_images_path + '/' + image
        #image_path=train_image_path+'/'+image
        #print("ok")
        store_inarray(image_path)
print(success)        


# In[9]:


store_image=np.array(store_image)
a,b,c=store_image.shape
store_image.resize(b,c,a)
store_image=(store_image-store_image.mean())/(store_image.std())
store_image=np.clip(store_image,0,1)
np.save('training1.npy',store_image)


# In[10]:


stae_model=Sequential()

stae_model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
stae_model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))
stae_model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))
stae_model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))
stae_model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
stae_model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

stae_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

stae_model.summary()


# In[11]:


training_data=np.load('training1.npy')
frames=training_data.shape[2]
frames=frames-frames%10

training_data=training_data[:,:,:frames]
training_data=training_data.reshape(-1,227,227,10)
training_data=np.expand_dims(training_data,axis=4)
target_data=training_data.copy()


# In[15]:




epochs=5
batch_size=1

callback_save = ModelCheckpoint("saved_model5.h5", monitor="mean_squared_error", save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

stae_model.fit(training_data,target_data, batch_size=batch_size, epochs=epochs, callbacks = [callback_save,callback_early_stopping])
stae_model.save("saved_model5.h5")


# In[ ]:


#threshold=0.0004244678
0.0003644678


# In[4]:


import cv2
import numpy as np 
from keras.models import load_model
import argparse
from PIL import Image
import imutils
#from google.colab.patches import cv2_imshow


def mean_squared_loss(x1,x2):
    
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples

    return mean_distance


model=load_model("saved_model5.h5")

cap = cv2.VideoCapture("C:/Users/80965/training_vid/testing_videos/05.avi")
print(cap.isOpened())


while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()
  

    for i in range(10):
        ret,frame=cap.read()
        
        image = imutils.resize(frame,width=700,height=600)

        frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
        gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
        gray=(gray-gray.mean())/gray.std()
        gray=np.clip(gray,0,1)
        
        imagedump.append(gray)
    
    imagedump=np.array(imagedump)
    

    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)
    
    output=model.predict(imagedump)
    

    loss=mean_squared_loss(imagedump,output)
    print("normal") 
    

    if frame.any()==None:
        
        print("none")

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    if loss>0.0003643678:
        #print("hi")
        print('Abnormal Event Detected')
        cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

    cv2.imshow("Video",image)
    

cap.release()
cv2.destroyAllWindows()


# In[ ]:




