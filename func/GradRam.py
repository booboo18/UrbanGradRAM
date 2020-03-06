#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020

GradRAM
simple function to run Regression Activation Map.
Adapted from GradCAM

@author: stephen.law
"""

import numpy as np
import keras
import pandas as pd
import cv2


from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.models import model_from_json

import keras.backend as K

def crop(img):
    min_side=min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
             centre[1]-min_side//2:centre[1]+min_side//2,:]
    return img

def flatten_model(model_nested):
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    model_flat = keras.models.Sequential(layers_flat)
    return model_flat

def load_img(path):
    X2 = image.load_img(path, target_size=(128, 128))
    X2 = image.img_to_array(X2)
    X2 = np.expand_dims(X2, axis=0)
    #X2 = X2-np.array([103.939, 116.779, 123.68])
    X2 = preprocess_input(X2, mode='tf')
    #X2 = preprocess_input(X2)
    img = np.array([i for i in X2])
    return img

def GradRAM(img,loaded_model):
    reg_output = loaded_model.output[:,:]
    last_conv_layer = loaded_model.get_layer("block5_conv3")
    grads = K.gradients(reg_output,last_conv_layer.get_output_at(1))[0]
    pooled_grads=K.mean(grads, axis=(0,1,2))
    iterate = K.function([loaded_model.input],[pooled_grads,last_conv_layer.get_output_at(1)[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(512):
        conv_layer_output_value[:,:,i]*=pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value,axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap/=np.max(heatmap)
    heatmap = cv2.resize(heatmap, (128,128))
    return heatmap
