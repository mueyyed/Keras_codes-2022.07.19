#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing data from MNIST  + how to save model with its parameters
import tensorflow as tf 
mnist = tf.keras.datasets.mnist


# In[14]:


#Divide Data
(x_train , y_train) , ( x_test , y_test) = mnist.load_data()


# In[16]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(x_train[222] , cmap= plt.cm.binary)
plt.show()

print(y_train[222])


# In[18]:


## normalization first way 
#x_train , x_test = x_train/255.0 , x_test/255.0 

# normalization second way 
x_train = tf.keras.utils.normalize(x_train , axis=1)
x_test = tf.keras.utils.normalize(x_test , axis=1)


print(x_train[0])
plt.imshow(x_train[0] , cmap=plt.cm.binary)
plt.show()


# In[19]:


## construct Model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),                             
    tf.keras.layers.Dense(128 , activation=tf.nn.relu), 
    tf.keras.layers.Dense(128 , activation=tf.nn.relu),
    tf.keras.layers.Dense(10 , activation=tf.nn.softmax)
])


# In[20]:


model.compile(optimizer ='adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])


# In[21]:


#Training model 
model.fit(x_train , y_train , epochs= 3) 


# In[23]:


model.evaluate(x_test , y_test)


# In[24]:


## Save model 

model.save('kerasNN.model')
new_model=tf.keras.models.load_model("kerasNN.model")
predictions = new_model.predict(x_test) 
print(predictions)


# In[25]:


import numpy as np 
print(np.argmax(predictions[0]))
plt.imshow(x_test[0] , cmap = plt.cm.binary)
plt.show()


# In[ ]:




