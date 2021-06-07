import os
import sys
import logging
import json
import copy
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import model_from_yaml, load_model, Model
from keras.layers import Input, Lambda
from keras.utils import to_categorical
import keras.backend as k
from keras.utils import Sequence

########### TODO: fix after install package #######################################################
from . import logging_daily
from . import configuration
from . import loss_and_metric
from . import readers
from . import samplers
from . import build_network
#####################################################################################################

class vggface_b_generator(Sequence):
    def __init__(self, reader, network, data_path, batch_size=1):
        self.reader = reader
        self.network = network
        self.batch_size = batch_size
        if reader.mode == 'train': self.ordered_idxs = np.load("%s/ordered_idxs.npy" % data_path, allow_pickle=True)
        else: self.ordered_idxs = np.load("%s/ordered_idxs_unknown.npy" % data_path, allow_pickle=True)
        self.steps = np.floor(self.ordered_idxs.shape[0] / float(batch_size)).astype(int) + 1
        self.feature_b = network.feature_b
        
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):        
        picked = np.concatenate(self.ordered_idxs[np.arange(idx*self.batch_size,
                                                            min((idx+1)*self.batch_size, self.ordered_idxs.shape[0]))])
        try:
            idxs = picked
            if self.feature_b:
                batch_idxs = idxs
                batch_feature_b = self.reader.all_features_for_b[batch_idxs]
                y = self.reader.class_to_int(self.reader.y[np.array(batch_idxs)])
            else:
                batch_imgs = []
                batch_idxs = []
                for i in idxs:
                    try:
                        batch_imgs.append(self.reader._read_vgg2face_image(self.reader.x_list[i]))
                        batch_idxs.append(i)
                    except Exception as e:
                        raise ValueError(e)
                batch_imgs = np.array(batch_imgs)
                batch_idxs = np.array(batch_idxs)
                y = self.reader.class_to_int(self.reader.y[np.array(batch_idxs)])
                if self.reader.augment:
                    batch_imgs = self.reader.get_augment(batch_imgs)
                if self.reader.normalize_sym:
                    batch_imgs = (batch_imgs - 0.5) * 2.
        except Exception as e:
            raise ValueError(e)
        
        b_noise = self.network.noise_sampler(y.shape[0], self.network.b_z_dim, self.network.b_sd)
        if self.feature_b: 
            xx = [batch_feature_b, y]
            yy = b_noise
        else:
            xx = [batch_imgs, y]
            yy = b_noise
        return xx, yy
#####################################################################################################

   
def extract_identity_feature(argdict, log):
    batch_size=int(argdict['batch_size'][0])
    try: reader_mode = argdict['mode'][0].strip()
    except: reader_mode = 'train'
        
    ### Configuration #####################################################################################
    config_data = configuration.Configurator(argdict['path_info'][0], log, verbose=False)
    config_data.set_config_map(config_data.get_section_map())
    config_network = configuration.Configurator(argdict['network_info'][0], log, verbose=False)
    config_network.set_config_map(config_network.get_section_map())

    path_info = config_data.get_config_map()
    network_info = config_network.get_config_map()

    architecture = path_info['model_info']['model_architecture']
    model_path = path_info['model_info']['model_dir']
    data_path = path_info['data_info']['data_path']
    
    ### Reader ###########################################################################################
    log.info('-----------------------------------------------------------------')
    log.info('Construct reader')
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())
    reader = reader_class(log, path_info, network_info, mode=reader_mode, verbose=True)

    ### Bulid network ####################################################################################
    log.info('-----------------------------------------------------------------')
    log.info('Build network')
    network_class = getattr(build_network, network_info['model_info']['network_class'].strip())  
    network = network_class(log, path_info, network_info, n_label=reader.get_n_label())
    network.build_model('./%s/%s' % (model_path,  path_info['model_info']['model_architecture']), verbose=2)
    network.load(model_path)
    log.info('-----------------------------------------------------------------')

    ### Sampler ##########################################################################################
    log.info('-----------------------------------------------------------------')
    log.info('Construct generator')
    generator = vggface_b_generator(reader, network, data_path, batch_size=batch_size)
    log.info('-----------------------------------------------------------------')

    ### Extract B ########################################################################################
    if network.feature_b:
        feature_for_b = Input(shape=network.feature_shape, name='feature_b_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        sample_b, b_given_x, b_j_given_x_j = network.encoder_b_model([feature_for_b, cls_info])
        get_b_model = Model([feature_for_b, cls_info], sample_b, name='get_b_model')
    else:
        real_image = Input(shape=network.image_shape, name='real_image_input', dtype='float32')
        cls_info = Input(shape=(1,), name='class_info_input', dtype='int32')
        sample_b, b_given_x, b_j_given_x_j = network.encoder_b_model([real_image, cls_info])
        get_b_model = Model([real_image, cls_info], sample_b, name='get_b_model')
    get_b_model.summary()

    all_b = get_b_model.predict_generator(generator, verbose=1)
    all_b = all_b[np.concatenate(generator.ordered_idxs)]

    log.info('Finished to extract all b of shape: %s' % str(all_b.shape))
    if reader_mode == 'train': 
        np.save('%s/all_b.npy' % model_path, all_b)
        log.info('Finished to save all b at %s/all_b.npy' % model_path)
    else:
        np.save('%s/all_b_unknown.npy' % model_path, all_b)
        log.info('Finished to save all b at %s/all_b_unknown.npy' % model_path)
    log.info('Computing End')
    log.info('-----------------------------------------------------------------')
