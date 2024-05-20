#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pydicom

from keras import layers
from keras.applications import DenseNet121, ResNet50V2, InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
from keras.utils import Sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout
from keras.initializers import glorot_normal, he_normal
from keras.regularizers import l2

import keras.metrics as M
import tensorflow_addons as tfa
import pickle

from keras import backend as K

import tensorflow as tf
from tensorflow.python.ops import array_ops

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold

import cupy as cp

import warnings
warnings.filterwarnings(action='once')


# In[ ]:





# In[3]:


BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = 'stage_2_train/'
TEST_DIR = 'stage_2_test/'


# In[4]:


train_df = pd.read_csv(BASE_PATH + 'stage_2_train.csv')
train_df['id'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1])
train_df['subtype'] = train_df['ID'].apply(lambda st: st.split('_')[2])
train_df.head()


# In[5]:


train_df = train_df[["id","subtype","Label"]]
train_df.head()


# In[6]:


train_df = pd.pivot_table(train_df,index="id",columns="subtype",values="Label")
train_df.head()


# In[7]:


pivot_df = train_df.copy()
pivot_df.drop("ID_6431af929",inplace=True)


# In[8]:


def map_to_gradient(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4 * grey_img - 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4 * grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4 * grey_img + 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    return rainbow_img

def rainbow_window(dcm):
    grey_img = window_image(dcm, 40, 80)
    return map_to_gradient(grey_img)

#import cupy as cp

def sigmoid_window(dcm, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    img = dcm.pixel_array
    img = cp.array(np.array(img))
    _, _, intercept, slope = get_windowing(dcm)
    img = img * slope + intercept
    ue = cp.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + cp.power(np.e, -1.0 * z))
    img = (img - cp.min(img)) / (cp.max(img) - cp.min(img))
    return cp.asnumpy(img)

def sigmoid_bsb_window(dcm):
    brain_img = sigmoid_window(dcm, 40, 80)
    subdural_img = sigmoid_window(dcm, 80, 200)
    bone_img = sigmoid_window(dcm, 600, 2000)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img

def window_image(dcm, window_center, window_width):
    _, _, intercept, slope = get_windowing(dcm)
    img = dcm.pixel_array * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    bone_img = window_image(dcm, 600, 2000)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


# In[9]:


def preprocess(file,type="WINDOW",fdir=TRAIN_DIR):
    dcm = pydicom.dcmread(BASE_PATH+fdir+file+".dcm")
    if type == "WINDOW":
        window_center , window_width, intercept, slope = get_windowing(dcm)
        w = window_image(dcm, window_center, window_width)
        win_img = np.repeat(w[:, :, np.newaxis], 3, axis=2)
        #return win_img
    elif type == "SIGMOID":
        window_center , window_width, intercept, slope = get_windowing(dcm)
        test_img = dcm.pixel_array
        w = sigmoid_window(dcm, window_center, window_width)
        win_img = np.repeat(w[:, :, np.newaxis], 3, axis=2)
        #return win_img
    elif type == "BSB":
        win_img = bsb_window(dcm)
        #return win_img
    elif type == "SIGMOID_BSB":
        win_img = sigmoid_bsb_window(dcm)
    elif type == "GRADIENT":
        win_img = rainbow_window(dcm)
        #return win_img
    else:
        win_img = dcm.pixel_array
    resized = cv2.resize(win_img,(224,224))
    return resized

class DataLoader(Sequence):
    def __init__(self, dataframe,
                 batch_size,
                 shuffle,
                 input_shape,
                 num_classes=6,
                 steps=None,
                 prep="BSB",
                 fdir=TRAIN_DIR):
        
        self.data_ids = dataframe.index.values
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.current_epoch=0
        self.prep = prep
        self.fdir = fdir
        self.steps=steps
        if self.steps is not None:
            self.steps = np.round(self.steps/3) * 3
            self.undersample()
        
    def undersample(self):
        part = np.int(self.steps/3 * self.batch_size)
        zero_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 0].index.values, size=5000, replace=False)
        hot_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 1].index.values, size=5000, replace=True)
        self.data_ids = list(set(zero_ids).union(hot_ids))
        np.random.shuffle(self.data_ids)
        
    # defines the number of steps per epoch
    def __len__(self):
        if self.steps is None:
            return np.int(np.ceil(len(self.data_ids) / np.float(self.batch_size)))
        else:
            return 3*np.int(self.steps/3) 
    
    # at the end of an epoch: 
    def on_epoch_end(self):
        # if steps is None and shuffle is true:
        if self.steps is None:
            self.data_ids = self.dataframe.index.values
            if self.shuffle:
                np.random.shuffle(self.data_ids)
        else:
            self.undersample()
        self.current_epoch += 1
    
    # should return a batch of images
    def __getitem__(self, item):
        # select the ids of the current batch
        current_ids = self.data_ids[item*self.batch_size:(item+1)*self.batch_size]
        X, y = self.__generate_batch(current_ids)
        return X, y
    
    # collect the preprocessed images and targets of one batch
    def __generate_batch(self, current_ids):
        X = np.empty((self.batch_size, *self.input_shape, 3))
        y = np.empty((self.batch_size, self.num_classes))
        for idx, ident in enumerate(current_ids):
            # Store sample
            #image = self.preprocessor.preprocess(ident) 
            image = preprocess(ident,self.prep,self.fdir)
            X[idx] = image
            # Store class
            y[idx] = self.__get_target(ident)
        return X, y
    
    # extract the targets of one image id:
    def __get_target(self, ident):
        targets = self.dataframe.loc[ident].values
        return targets


# In[10]:


def DenseNet():
    densenet = DenseNet121(
    #weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    weights='imagenet',
    include_top=False)
    return densenet
def ResNet():
    resnet = ResNet50V2(weights="imagenet",include_top=False)
    return resnet
def Inception():
    incept = InceptionV3(weights="imagenet",include_top=False)
    return incept

def get_backbone(name):
    if name == "RESNET":
        return ResNet
    elif name == "DENSE":
        return DenseNet
    elif name == "INCEPT":
        return Inception

def build_model(backbone):
    m = backbone()
    x = m.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.3)(x)
    pred = Dense(6,activation="sigmoid")(x)
    model = Model(inputs=m.input,outputs=pred)
    return model


# In[11]:


train,test = train_test_split(pivot_df,test_size=0.2,random_state=42,shuffle=True)

split_seed = 1
kfold = StratifiedKFold(n_splits=5, random_state=split_seed,shuffle=True).split(np.arange(train.shape[0]), train["any"].values)

train_idx, dev_idx = next(kfold)

train_data = train.iloc[train_idx]
dev_data = train.iloc[dev_idx]

print(train_data.shape)
print(dev_data.shape)


# In[12]:


f1 = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
def casting_focal_loss():
    def inner_casting(y_true,y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.clip_by_value(y_true,1e-7,1-1e-7)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred,1e-7,1-1e-7)
        
        return f1(y_true,y_pred)
    return inner_casting
METRICS = ['categorical_accuracy']

#LOSS = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)
LOSS = casting_focal_loss()

BATCH_SIZE = 32
TRAIN_STEPS = 500#train_data.shape[0] // BATCH_SIZE
VAL_STEPS = 500#dev_data.shape[0] // BATCH_SIZE
EPOCHS = 10
#WEIGHT = [2.0,1.0,1.0,1.0,1.0,1.0]
ALPHA = 0.5
GAMMA = 2

LR = 0.0001

PREP = "SIGMOID"
ARCH = 'RESNET'

train_dataloader = DataLoader(train_data,
                              BATCH_SIZE,
                              shuffle=True,
                              input_shape=(224,224),
                              steps=TRAIN_STEPS,
                              prep=PREP)

dev_dataloader = DataLoader(dev_data, 
                            BATCH_SIZE,
                            shuffle=True,
                            input_shape=(224,224),
                            steps=VAL_STEPS,
                            prep=PREP)
test_dataloader = DataLoader(test,
                            BATCH_SIZE,
                            shuffle=False,
                            input_shape=(224,224),
                            prep=PREP)

cpath = "./" + ARCH + "_" + PREP + "_" + str(TRAIN_STEPS) + "_" + str(EPOCHS)
checkpoint = ModelCheckpoint(filepath=cpath + ".model",mode="min",verbose=1,save_best_only=True,save_weights_only=False,period=1)

model = build_model(get_backbone(ARCH))

model.compile(optimizer=Adam(learning_rate=LR),loss=LOSS,metrics=METRICS)

history = model.fit_generator(generator=train_dataloader,validation_data=dev_dataloader,epochs=EPOCHS,workers=8,callbacks=[checkpoint])

with open(cpath + ".history", 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
print("Generating predictions")


# In[13]:


test_csv = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"
BASE_PATH = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"
TEST_DIR = "stage_2_test/"
test_df = pd.read_csv(test_csv)
test_df.head()


# In[14]:


testdf = test_df.ID.str.rsplit("_", n=1, expand=True)
testdf = testdf.rename({0: "id", 1: "subtype"}, axis=1)
testdf.loc[:, "label"] = 0
testdf.head()


# In[15]:


testdf = pd.pivot_table(testdf, index="id", columns="subtype", values="label")
testdf.head()


# In[16]:


def turn_pred_to_dataframe(data_df, pred):
    df = pd.DataFrame(pred, columns=data_df.columns, index=data_df.index)
    df = df.stack().reset_index()
    df.loc[:, "ID"] = df.id.str.cat(df.subtype, sep="_")
    df = df.drop(["id", "subtype"], axis=1)
    df = df.rename({0: "Label"}, axis=1)
    return df


# In[17]:


test_dataloader = DataLoader(testdf,32,shuffle=False,input_shape=(224,224),prep="SIGMOID",fdir=TEST_DIR)


# In[18]:


test_pred = model.predict(test_dataloader,verbose=1)


# In[20]:


pred = test_pred[0:testdf.shape[0]]
pred_df = turn_pred_to_dataframe(testdf,pred)
pred_df.to_csv("resnet_mfl_pred.csv",index=False)

