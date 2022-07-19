#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import os 
import cv2 
from tqdm import tqdm 
import tensorflow as tf 
from tensorflow import keras


# In[6]:


# prepare data 
DATADIR = 'PetImages'
CATEGORIES = ['Dog'  , 'Cat']

for category in CATEGORIES : # do dogs and cats
    path = os.path.join(DATADIR , category) # create path to dogs and cats
    x=0 
    for img in os.listdir(path):# iterate over each image per dogs and cats
        x+=1 
        img_array = cv2.imread(os.path.join(path, img) , cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array , cmap = 'gray')# graph it 
        plt.show() 
        if x==10: #display 
            break
print(img_array)
print(img_array.shape) 


# In[9]:


IMG_SIZE = 50 
new_array = cv2.resize(img_array , (IMG_SIZE , IMG_SIZE))
plt.imshow(new_array , cmap= 'gray')
plt.show()


# In[11]:


# Resizing Data 
training_data=[]

def create_training_data():
    for category in CATEGORIES: 
        path=os.path.join(DATADIR , category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category) # get classification( 0 or 1 )
        
        for img in tqdm (os.listdir(path)): # iterate over img per dog and cats
            try:
                img_array= cv2.imread(os.path.join(path , img) , cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array , (IMG_SIZE , IMG_SIZE))  # resize to nor 
                training_data.append([new_array , class_num])  #add this to our training 
            except Exception as e: # to keep output clean 
                pass

create_training_data() 
print(len(training_data))


# In[12]:


# make randomization + shuffle 
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])


# In[13]:


# reshape data
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)


# In[14]:


import pickle 
pickle_out =open("X.pickle" , "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle" , "wb")
pickle.dump(y , pickle_out)
pickle_out.close()

pickle_in = open("X.pickle" , "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle" , "rb")
y = pickle.load(pickle_in)


# In[15]:


# contructing model  + customize + training
model = keras.Sequential( [
    keras.layers.Flatten(input_shape=(IMG_SIZE , IMG_SIZE)),
    keras.layers.Dense(128 , activation = tf.nn.sigmoid) , 
    keras.layers.Dense(128 , activation = tf.nn.sigmoid) ,
    keras.layers.Dense(2 , activation = tf.nn.softmax) 
])


# In[16]:


model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])


# In[ ]:


model.fit( x , y , epochs=3)


# In[ ]:


# evaluate data 
test_loss , test_acc = model.evaluate(X,y)
test_acc

