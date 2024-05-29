#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import time


# In[ ]:


GAP=tf.keras.layers.GlobalAveragePooling1D()


# In[ ]:


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
        return K_XX, K_XY, K_YY, tf.reduce_sum(wts)
    

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    return mmd2


# In[ ]:


optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
fe_optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
classification_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)  
recon_loss=tf.keras.losses.MeanSquaredError()

def mmd(X1,X2, bandwidths=[5]):
    kernel_loss = mix_rbf_mmd2(X1,X2, sigmas=bandwidths)
    return kernel_loss*5


def information_entropy(x):
    b=[]
    a=tf.nn.softmax(x,axis=-1)
    for i in range(a.shape[0]):
        for j in a[i]:
            log_j=tf.math.log(tf.clip_by_value(j,1e-32,tf.reduce_max(j)))/tf.math.log(2.)
            j=j
            b.append(-j*log_j)
    b=tf.reduce_sum(b,axis=-1)
    return -b


def mine_loss(model,x,y):
    batch_size=x.shape[0]
    tiled_x=tf.concat([x,x],axis=0)
    #shuffled_y=y
    shuffled_y=tf.random.shuffle(y)
    concat_y=tf.concat([y,shuffled_y],axis=0)
    inputs=tf.concat([tiled_x,concat_y],axis=1)
    logits=model(inputs)

    pred_xy=logits[:batch_size]
    pred_x_y=logits[batch_size:]
    loss = - tf.experimental.numpy.log2(tf.experimental.numpy.exp(1)) * (tf.reduce_mean(pred_xy) - tf.math.log(tf.reduce_mean(tf.math.exp(pred_x_y))))
    return loss


# In[ ]:


# for class_train_step
class_cr_loss1=tf.keras.metrics.Mean()
class_cr_loss2=tf.keras.metrics.Mean()
class_cr_loss3=tf.keras.metrics.Mean()
class_cr_loss4=tf.keras.metrics.Mean()
class_ci_loss1=tf.keras.metrics.Mean()
class_ci_loss2=tf.keras.metrics.Mean()
class_ci_loss3=tf.keras.metrics.Mean()
class_ci_loss4=tf.keras.metrics.Mean()
class_cr_acc1 = tf.keras.metrics.CategoricalAccuracy()
class_cr_acc2 = tf.keras.metrics.CategoricalAccuracy()
class_cr_acc3 = tf.keras.metrics.CategoricalAccuracy()
class_cr_acc4 = tf.keras.metrics.CategoricalAccuracy()    
class_ci_acc1 = tf.keras.metrics.CategoricalAccuracy()
class_ci_acc2 = tf.keras.metrics.CategoricalAccuracy()
class_ci_acc3 = tf.keras.metrics.CategoricalAccuracy()
class_ci_acc4 = tf.keras.metrics.CategoricalAccuracy()
cr_loss1,cr_loss2,cr_loss3,cr_loss4,ci_loss1,ci_loss2,ci_loss3,ci_loss4=[],[],[],[],[],[],[],[]
cr_acc1,cr_acc2,cr_acc3,cr_acc4,ci_acc1,ci_acc2,ci_acc3,ci_acc4=[],[],[],[],[],[],[],[]


#for mi_re_train_step
dis_re_loss1=tf.keras.metrics.Mean()
dis_re_loss2=tf.keras.metrics.Mean()
dis_re_loss3=tf.keras.metrics.Mean()
dis_re_loss4=tf.keras.metrics.Mean()    
dis_mi_loss1=tf.keras.metrics.Mean()
dis_mi_loss2=tf.keras.metrics.Mean()
dis_mi_loss3=tf.keras.metrics.Mean()
dis_mi_loss4=tf.keras.metrics.Mean()
re_loss1,re_loss2,re_loss3,re_loss4,mi_loss1,mi_loss2,mi_loss3,mi_loss4=[],[],[],[],[],[],[],[]


# for dc_train_step
dc_mmd_loss1=tf.keras.metrics.Mean()
dc_mmd_loss2=tf.keras.metrics.Mean()
dc_mmd_loss3=tf.keras.metrics.Mean()
dc_mmd_loss4=tf.keras.metrics.Mean()
dc_mmd_loss5=tf.keras.metrics.Mean()
dc_mmd_loss6=tf.keras.metrics.Mean()
dc_ie_loss = tf.keras.metrics.Mean()
mmd_loss1,mmd_loss2,mmd_loss3,mmd_loss4,mmd_loss5,mmd_loss6,ie_loss=[],[],[],[],[],[],[]


# In[ ]:


def class_test_step(feature_extractor,disentangler,classifier,data1,data2,data3,data4,label1,label2,label3,label4):
    with tf.GradientTape(persistent=True) as t:
        fe1= feature_extractor(data1,training = True)
        fe2= feature_extractor(data2,training = True)
        fe3= feature_extractor(data3,training = True)
        fe4= feature_extractor(data4,training = True)
        
        ci1,cr1= disentangler(fe1,training = True)
        ci2,cr2= disentangler(fe2,training = True)
        ci3,cr3= disentangler(fe3,training = True)
        ci4,cr4= disentangler(fe4,training = True)
        
        pre_cr1=classifier(GAP(cr1),training = True)
        pre_cr2=classifier(GAP(cr2),training = True)
        pre_cr3=classifier(GAP(cr3),training = True)
        pre_cr4=classifier(GAP(cr4),training = True)

        pre_ci1=classifier(GAP(ci1),training = True)
        pre_ci2=classifier(GAP(ci2),training = True)
        pre_ci3=classifier(GAP(ci3),training = True)
        pre_ci4=classifier(GAP(ci4),training = True)

        cr_loss1 = classification_loss(label1,pre_cr1)
        cr_loss2 = classification_loss(label2,pre_cr2)
        cr_loss3 = classification_loss(label3,pre_cr3)
        cr_loss4 = classification_loss(label4,pre_cr4)
        
        ci_loss1 = classification_loss(label1,pre_ci1)
        ci_loss2 = classification_loss(label2,pre_ci2)
        ci_loss3 = classification_loss(label3,pre_ci3)
        ci_loss4 = classification_loss(label4,pre_ci4)
        
        new_label=tf.ones_like(label1)/label1.shape[-1]
        ie_loss1 = classification_loss(new_label,pre_ci1)
        ie_loss2 = classification_loss(new_label,pre_ci2)
        ie_loss3 = classification_loss(new_label,pre_ci3)
        ie_loss4 = classification_loss(new_label,pre_ci4)
        
    w1=feature_extractor.variables
    w2=disentangler.variables
    w3=classifier.variables
    
    # extract the class_relevant features by supervised training 
    fe_cr_grads1 = t.gradient(cr_loss1, w1)
    fe_optimizer.apply_gradients(zip(fe_cr_grads1, w1))
    cr_grads1 = t.gradient(cr_loss1, w2+w3)
    optimizer.apply_gradients(zip(cr_grads1, w2+w3))
    
    #extract the class-irrelevant features by adversarial training
    ci_grads1 = t.gradient(ci_loss1, w2+w3)
    optimizer.apply_gradients(zip(ci_grads1, w2+w3))
    
    ie_grads1 = t.gradient(ie_loss1, w2)
    optimizer.apply_gradients(zip(ie_grads1, w2))
    
    # evaluation metric collection
    class_cr_loss1(cr_loss1)
    class_cr_loss2(cr_loss2)
    class_cr_loss3(cr_loss3)
    class_cr_loss4(cr_loss4)
    class_ci_loss1(ci_loss1)
    class_ci_loss2(ci_loss2)
    class_ci_loss3(ci_loss3)
    class_ci_loss4(ci_loss4)
    class_cr_acc1(label1,pre_cr1)
    class_cr_acc2(label2,pre_cr2)
    class_cr_acc3(label3,pre_cr3)
    class_cr_acc4(label4,pre_cr4)
    class_ci_acc1(label1,pre_ci1)
    class_ci_acc2(label2,pre_ci2)
    class_ci_acc3(label3,pre_ci3)
    class_ci_acc4(label4,pre_ci4)    
    
    return fe1,pre_cr1
    
    
def mi_re_test_step(feature_extractor,disentangler,reconstructor,net,data1,data2,data3,data4,label1,label2,label3,label4):
    with tf.GradientTape(persistent=True) as t:
        fe1= feature_extractor(data1,training = True)
        fe2= feature_extractor(data2,training = True)
        fe3= feature_extractor(data3,training = True)
        fe4= feature_extractor(data4,training = True)
        
        ci1,cr1= disentangler(fe1,training = True)
        ci2,cr2= disentangler(fe2,training = True)
        ci3,cr3= disentangler(fe3,training = True)
        ci4,cr4= disentangler(fe4,training = True)
            
        re1= reconstructor([ci1,cr1],training = True)
        re2= reconstructor([ci2,cr2],training = True)
        re3= reconstructor([ci3,cr3],training = True)
        re4= reconstructor([ci4,cr4],training = True)
        
        re_loss1= recon_loss(fe1,re1)
        re_loss2= recon_loss(fe2,re2)
        re_loss3= recon_loss(fe3,re3)
        re_loss4= recon_loss(fe4,re4)
        
        '''
        mi_loss1=mine_loss(net,GAP(ci1),GAP(cr1))
        mi_loss2=mine_loss(net,GAP(ci2),GAP(cr2))
        mi_loss3=mine_loss(net,GAP(ci3),GAP(cr3))
        '''
        
        mi_loss1=-mmd(GAP(ci1),GAP(cr1))
        mi_loss2=-mmd(GAP(ci2),GAP(cr2))
        mi_loss3=-mmd(GAP(ci3),GAP(cr3))        
        mi_loss4=-mmd(GAP(ci4),GAP(cr4))        
        
    w1=feature_extractor.variables
    w2=disentangler.variables
    w3=net.variables
    w4=reconstructor.variables
    #print("111",w1[0],"222",w2[0],"333",w3[0],"444",w4[0])
    
    # separate the class_irrelevant features and the class_rrelevant features    
    max_grads1 = t.gradient(mi_loss1, w2)
    optimizer.apply_gradients(zip(max_grads1, w2))     
    
    # reconstructe the class_irrelevant features and the class_rrelevant features
    re_grads1 = t.gradient(re_loss1, w2+w4)
    optimizer.apply_gradients(zip(re_grads1, w2+w4))

    
    # evaluation metric collection
    dis_re_loss1(re_loss1)
    dis_re_loss2(re_loss2)
    dis_re_loss3(re_loss3)
    dis_re_loss4(re_loss4)
    dis_mi_loss1(-mi_loss1)
    dis_mi_loss2(-mi_loss2)
    dis_mi_loss3(-mi_loss3)
    dis_mi_loss4(-mi_loss4)
    

def dc_test_step(feature_extractor,disentangler,data1,data2,data3,data4,label1,label2,label3,label4):
    with tf.GradientTape(persistent=True) as t:
        fe1= feature_extractor(data1,training = True)
        fe2= feature_extractor(data2,training = True)
        fe3= feature_extractor(data3,training = True)
        fe4= feature_extractor(data4,training = True)
        
        ci1,cr1= disentangler(fe1,training = True)
        ci2,cr2= disentangler(fe2,training = True)
        ci3,cr3= disentangler(fe3,training = True)
        ci4,cr4= disentangler(fe4,training = True)
        
        dc_loss1=10*mmd(GAP(cr1),GAP(cr2))
        dc_loss2=10*mmd(GAP(cr2),GAP(cr3))
        dc_loss3=10*mmd(GAP(cr1),GAP(cr3))
        dc_loss4=10*mmd(GAP(cr1),GAP(cr4))
        dc_loss5=10*mmd(GAP(cr2),GAP(cr4))
        dc_loss6=10*mmd(GAP(cr3),GAP(cr4))
        adc_loss=classification_loss([1/6,1/6,1/6,1/6,1/6,1/6],[dc_loss1/0.2,dc_loss2/0.2,dc_loss3/0.2,dc_loss4/0.2,dc_loss5/0.2,dc_loss6/0.2])
        
    w1=feature_extractor.variables
    w2=disentangler.variables
    # separate the class_irrelevant features and the class_rrelevant features

    # evaluation metric collection
    dc_mmd_loss1(dc_loss1)
    dc_mmd_loss2(dc_loss2)
    dc_mmd_loss3(dc_loss3)
    dc_mmd_loss4(dc_loss4)
    dc_mmd_loss5(dc_loss5)
    dc_mmd_loss6(dc_loss6)
    dc_ie_loss(adc_loss) 
    
    
def test (feature_extractor,disentangler,reconstructor,classifier,net,all_number,class_number,mi_re_number,dc_number,dataset,print_gap):
    for all_epoch in range(all_number):
        for class_epoch in range (class_number):
            for (batch, (data1,data2,data3,data4,label1,label2,label3,label4)) in enumerate(dataset):
                orig_feature,label_feature=class_test_step(feature_extractor,disentangler,classifier,data1,data2,data3,data4,label1,label2,label3,label4)
                if class_epoch==class_number-1 and all_epoch==all_number-1:
                    if batch==0:
                        init_orig_feature=orig_feature
                        init_label_feature=label_feature
                    else:
                        init_orig_feature=tf.concat([init_orig_feature,orig_feature],axis=0)
                        init_label_feature=tf.concat([init_label_feature,label_feature],axis=0)

            cr_loss1.append(class_cr_loss1.result()),cr_loss2.append(class_cr_loss2.result()),cr_loss3.append(class_cr_loss3.result()),cr_loss4.append(class_cr_loss4.result())
            ci_loss1.append(class_ci_loss1.result()),ci_loss2.append(class_ci_loss2.result()),ci_loss3.append(class_ci_loss3.result()),ci_loss4.append(class_ci_loss4.result())
            cr_acc1.append(class_cr_acc1.result()),cr_acc2.append(class_cr_acc2.result()),cr_acc3.append(class_cr_acc3.result()),cr_acc4.append(class_cr_acc4.result())
            ci_acc1.append(class_ci_acc1.result()),ci_acc2.append(class_ci_acc2.result()),ci_acc3.append(class_ci_acc3.result()),ci_acc4.append(class_ci_acc4.result())
            
            class_cr_loss1.reset_states(),class_cr_loss2.reset_states(),class_cr_loss3.reset_states(),class_cr_loss4.reset_states()
            class_ci_loss1.reset_states(),class_ci_loss2.reset_states(),class_ci_loss3.reset_states(),class_ci_loss4.reset_states()
            class_cr_acc1.reset_states(),class_cr_acc2.reset_states(),class_cr_acc3.reset_states(),class_cr_acc4.reset_states()            
            class_ci_acc1.reset_states(),class_ci_acc2.reset_states(),class_ci_acc3.reset_states(),class_ci_acc4.reset_states()
            
        for mi_re_epoch in range (mi_re_number):
            for (batch, (data1,data2,data3,data4,label1,label2,label3,label4)) in enumerate(dataset):
                mi_re_test_step(feature_extractor,disentangler,reconstructor,net,data1,data2,data3,data4,label1,label2,label3,label4)
            re_loss1.append(dis_re_loss1.result()),re_loss2.append(dis_re_loss2.result()),re_loss3.append(dis_re_loss3.result()),re_loss4.append(dis_re_loss4.result())
            mi_loss1.append(dis_mi_loss1.result()),mi_loss2.append(dis_mi_loss2.result()),mi_loss3.append(dis_mi_loss3.result()),mi_loss4.append(dis_mi_loss4.result())
            
            dis_re_loss1.reset_states(),dis_re_loss2.reset_states(),dis_re_loss3.reset_states(),dis_re_loss4.reset_states()
            dis_mi_loss1.reset_states(),dis_mi_loss2.reset_states(),dis_mi_loss3.reset_states(),dis_mi_loss4.reset_states()
        
        for dc_epoch in range (dc_number):
            for (batch, (data1,data2,data3,data4,label1,label2,label3,label4)) in enumerate(dataset):
                dc_test_step(feature_extractor,disentangler,data1,data2,data3,data4,label1,label2,label3,label4)
            mmd_loss1.append(dc_mmd_loss1.result()),mmd_loss2.append(dc_mmd_loss2.result())
            mmd_loss3.append(dc_mmd_loss3.result()),mmd_loss4.append(dc_mmd_loss4.result())
            mmd_loss5.append(dc_mmd_loss5.result()),mmd_loss6.append(dc_mmd_loss6.result()),ie_loss.append(dc_ie_loss.result())
            
            dc_mmd_loss1.reset_states(),dc_mmd_loss2.reset_states(),dc_mmd_loss3.reset_states()
            dc_mmd_loss4.reset_states(),dc_mmd_loss5.reset_states(),dc_mmd_loss6.reset_states(),dc_ie_loss.reset_states()
        
        
        if (all_epoch+1)%print_gap==0:
            print('epoch:{},\n,cr_loss1:{:.5f},cr_loss2:{:.5f},cr_loss3:{:.5f},cr_loss4:{:.5f},ci_loss1:{:.5f},ci_loss2:{:.5f},ci_loss3:{:.5f},ci_loss4:{:.5f},\n,cr_acc1:{:.5f},cr_acc2:{:.5f},cr_acc3:{:.5f},cr_acc4:{:.5f},ci_acc1:{:.5f},ci_acc2:{:.5f},ci_acc3:{:.5f},ci_acc4:{:.5f}'.format(all_epoch+1,
                            cr_loss1[class_number*(all_epoch+1)-1],cr_loss2[class_number*(all_epoch+1)-1],cr_loss3[class_number*(all_epoch+1)-1],cr_loss4[class_number*(all_epoch+1)-1],
                            ci_loss1[class_number*(all_epoch+1)-1],ci_loss2[class_number*(all_epoch+1)-1],ci_loss3[class_number*(all_epoch+1)-1],ci_loss4[class_number*(all_epoch+1)-1],
                            cr_acc1[class_number*(all_epoch+1)-1],cr_acc2[class_number*(all_epoch+1)-1],cr_acc3[class_number*(all_epoch+1)-1],cr_acc4[class_number*(all_epoch+1)-1],
                            ci_acc1[class_number*(all_epoch+1)-1],ci_acc2[class_number*(all_epoch+1)-1],ci_acc3[class_number*(all_epoch+1)-1],ci_acc4[class_number*(all_epoch+1)-1]))       
            print('re_loss1:{:.5f},re_loss2:{:.5f},re_loss3:{:.5f},re_loss4:{:.5f},mi_loss1:{:.5f},mi_loss2:{:.5f},mi_loss3:{:.5f},mi_loss4:{:.5f}'.format(
                            re_loss1[mi_re_number*(all_epoch+1)-1],re_loss2[mi_re_number*(all_epoch+1)-1],re_loss3[mi_re_number*(all_epoch+1)-1],re_loss4[mi_re_number*(all_epoch+1)-1],
                            mi_loss1[mi_re_number*(all_epoch+1)-1],mi_loss2[mi_re_number*(all_epoch+1)-1],mi_loss3[mi_re_number*(all_epoch+1)-1],mi_loss4[mi_re_number*(all_epoch+1)-1]))            
            print('mmd_loss1:{:.5f},mmd_loss2:{:.5f},mmd_loss3:{:.5f},mmd_loss4:{:.5f},mmd_loss5:{:.5f},mmd_loss6:{:.5f},ie_loss:{:.5f}'.format(
                            mmd_loss1[dc_number*(all_epoch+1)-1],mmd_loss2[dc_number*(all_epoch+1)-1],mmd_loss3[dc_number*(all_epoch+1)-1],
                            mmd_loss4[dc_number*(all_epoch+1)-1],mmd_loss5[dc_number*(all_epoch+1)-1],mmd_loss6[dc_number*(all_epoch+1)-1],ie_loss[dc_number*(all_epoch+1)-1]))
            
    all_loss=["cr_loss1","cr_loss2","cr_loss3","cr_loss4",
              "ci_loss1","ci_loss2","ci_loss3","ci_loss4",
              "cr_acc1","cr_acc2","cr_acc3","cr_acc4",
              "ci_acc1","ci_acc2","ci_acc3","ci_acc4",
              "re_loss1","re_loss2","re_loss3","re_loss4",
              "mi_loss1","mi_loss2","mi_loss3","mi_loss4",
              "mmd_loss1","mmd_loss2","mmd_loss3",
              "mmd_loss4","mmd_loss5","mmd_loss6","ie_loss"]
    df={}
    for i in all_loss:
        df[i]=np.array(eval(i))
    return df,init_orig_feature,init_label_feature


# In[ ]:




