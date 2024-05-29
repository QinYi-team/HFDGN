#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


# In[2]:


class Net(tf.keras.Model):
    def __init__(self,x_dim,y_dim) :
        super((Net), self).__init__()
        self.lay = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(x_dim+y_dim,),activation='relu'),
            tf.keras.layers.Dense(4,activation='relu'),
            tf.keras.layers.Dense(1),])
    def call(self,x,y):
        batch_size=x.shape[0]
        tiled_x=tf.concat([x,x],axis=0)
        
        shuffled_y=tf.random.shuffle(y)
        concat_y=tf.concat([y,shuffled_y],axis=0)
        inputs=tf.concat([tiled_x,concat_y],axis=1)
        logits=self.lay(inputs)
        
        pred_xy=logits[:batch_size]
        pred_x_y=logits[batch_size:]
        loss = - tf.experimental.numpy.log2(tf.experimental.numpy.exp(1)) * (tf.reduce_mean(pred_xy) - tf.math.log(tf.reduce_mean(tf.math.exp(pred_x_y))))
        return loss
    
class Estimator():
    def __init__(self,x_dim,y_dim) -> None:
        self.net=Net(x_dim,y_dim)
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

    def backward(self,x,y,epoch):
        MI=[]
        for i in range (epoch):
            with tf.GradientTape() as t:
                loss=self.net(x,y)
            grads = t.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
            MI.append(-loss)
        return MI


# In[ ]:


if __name__ == "__main__":
    #定义数据，初始化网络，并定义采样函数
    power = 3
    noise = 0.8
    n_epoch = 512
    batch_size = 512
    x_dim = 128
    y_dim = 128


    estimator = Estimator(x_dim,y_dim)
    def gen_x(num, dim ,power):
        return np.random.normal(0., np.sqrt(power), [num, dim])


    def gen_y(x, num, dim,noise):
        return x[:,:dim] + np.random.normal(0., np.sqrt(noise), [num, dim])


    def true_mi(power, noise, dim):
        return dim * 0.5 * np.log2(1 + power/noise)


    #互信息真实值
    mi = true_mi(power, noise, y_dim)
    print('True MI:', mi)


    #开始训练，生成数据，反向传播
    x_sample = gen_x(batch_size, x_dim, power)
    y_sample = gen_y(x_sample, batch_size, y_dim ,noise)

    x_sample = tf.convert_to_tensor(x_sample,dtype="float32")
    y_sample = tf.convert_to_tensor(y_sample,dtype="float32")

    info = estimator.backward(x_sample,y_sample,2000)
    plt.plot(info)


# In[15]:


if __name__ == "__main__":    
    def NET():
        model=tf.keras.Sequential([
                tf.keras.layers.Dense(8, input_shape=(256,),activation='relu'),
                tf.keras.layers.Dense(4,activation='relu'),
                tf.keras.layers.Dense(1),])
        return model
    model=NET()

    def mine_loss(model,x,y):
        batch_size=x.shape[0]
        tiled_x=tf.concat([x,x],axis=0)

        shuffled_y=tf.random.shuffle(y)
        concat_y=tf.concat([y,shuffled_y],axis=0)
        inputs=tf.concat([tiled_x,concat_y],axis=1)
        logits=model(inputs)

        pred_xy=logits[:batch_size]
        pred_x_y=logits[batch_size:]
        loss = - tf.experimental.numpy.log2(tf.experimental.numpy.exp(1)) * (tf.reduce_mean(pred_xy) - tf.math.log(tf.reduce_mean(tf.math.exp(pred_x_y))))
        return loss

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    MI=[]
    for i in range (2000):
        with tf.GradientTape() as t:
            loss=mine_loss(model,x_sample,y_sample)
        grads = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        MI.append(-loss)


# In[ ]:




