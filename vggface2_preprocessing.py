import os
import sys
import logging
import numpy as np
import copy
import random

import tensorflow as tf

seed_value = 123
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

import keras.backend as k
from keras.utils import Sequence

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
k.set_session(tf.Session(config=config))

sys.path.append('/'.join(os.getcwd().split('/')))
from ornstein_auto_encoder import logging_daily
from ornstein_auto_encoder import readers
from ornstein_auto_encoder.utils import argv_parse

### Load pretrained VGGFace2 Classifier
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

class vggface_ebmedding_generator(Sequence):
    def __init__(self, reader, batch_size=1, target_size=(224, 224)):
        self.reader = reader
        self.batch_size = batch_size
        self.target_size = target_size
        self.steps = np.floor(reader.x_list.shape[0] / float(batch_size)).astype(int) + 1
        self.img_path_list = np.array(['%s/%s/%s' % (reader.data_path, reader.mode, img_path) for img_path in reader.x_list])
        
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        picked = np.arange(idx*self.batch_size, min((idx+1)*self.batch_size, self.img_path_list.shape[0]))
        x = np.array([self.get_preprocessed_input(img_path) for img_path in self.img_path_list[picked]])
        y = self.reader.y[picked]
        return x, y
    
    def get_preprocessed_input(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        x = utils.preprocess_input(x, version=2) # VGGFace2 for SENET50
        return x


if __name__=='__main__':
    argdict = argv_parse(sys.argv)
    logger = logging_daily.logging_daily(argdict['log_info'][0])
    logger.reset_logging()
    log = logger.get_logging()
    log.setLevel(logging_daily.logging.INFO)
    
    log.info('----------------------------------------------------------------------------------------')
    log.info('Preprocessing for VGGFace2')
    log.info('----------------------------------------------------------------------------------------')
    
    data_path = argdict['data_dir'][0]

    ### Set basic variables
    path_info = {'data_info': {'data_path': data_path},
                 'model_info': {'train_tot_idxs': 'train_tot_idxs.npy',
                                'test_tot_idxs': 'test_tot_idxs.npy'}}

    network_info = {'model_info': {'reader_class': 'VGGFace2Reader',
                                   'n_label': 'None',
                                   'normalizer': 'normalize_none',
                                   'normalize_sym': 'True',
                                   'crop_style': 'closecrop',
                                   'augment': 'False',
                                   'except_class': '',
                                   'minarity_group_size': 'None',
                                   'minarity_ratio': 'None',
                                   'img_shape': '128, 128, 3'},
                    'training_info': {'sampler_class': 'EpochClassSampler',
                                      'sampler_class_per_epoch': '60',
                                      'sampler_class_per_batch': '10',
                                      'sampler_decay': '1.',
                                      'batch_size': '600',
                                      'sequential': 'False',
                                      'replace': 'False'},
                    'validation_info': {'sampler_class': 'BatchClassSampler',
                                      'sampler_class_per_batch': '10',
                                      'sampler_decay': '1.',
                                      'batch_size': '100',
                                      'sequential': 'False',
                                      'replace': 'False'}}
    
    vgg_model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Identities in the training set (known)
    ## Reader
    reader_class = getattr(readers, network_info['model_info']['reader_class'].strip())  
    reader = reader_class(log, path_info, network_info, mode='train', verbose=True)
    
    ordered_idxs = np.array([np.where(reader.y==y)[0] for y in reader.y_class])
    np.save("%s/ordered_idxs.npy" % data_path, ordered_idxs, allow_pickle=True)
        
    vgg_generator = vggface_ebmedding_generator(reader, batch_size=600)
    
    all_features = vgg_model.predict_generator(vgg_generator, verbose=1)
    # np.save('all_features.npy', all_features)

    all_features_normalized = all_features / np.sqrt(np.sum(np.square(all_features), axis=1, keepdims=True))
    np.save('%s/all_features_normalized.npy' % data_path, all_features_normalized)
    log.info('all_features_normalized.npy saved')
    
    # Identities in the test set (unknown)
    new_network_info = copy.deepcopy(network_info)
    new_path_info = copy.deepcopy(path_info)
    new_reader = reader_class(log, new_path_info, new_network_info, mode='test', verbose=True)
    
    new_ordered_idxs = np.array([np.where(new_reader.y==y)[0] for y in new_reader.y_class])
    np.save("%s/ordered_idxs_unknown.npy" % data_path, new_ordered_idxs, allow_pickle=True)
    
    new_vgg_generator = vggface_ebmedding_generator(new_reader, batch_size=600)
    all_features_of_unknown = vgg_model.predict_generator(new_vgg_generator, verbose=1)
    
    all_features_of_unknown_normalized = all_features_of_unknown / np.sqrt(np.sum(np.square(all_features_of_unknown), axis=1, keepdims=True))
    np.save('%s/all_features_of_unknown_normalized.npy' % data_path, all_features_of_unknown_normalized)
    log.info('all_features_of_unknown_normalized.npy saved')
    
    log.info('Preprocessing finished')
    log.info('----------------------------------------------------------------------------------------')