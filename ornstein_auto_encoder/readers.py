import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

import PIL
from PIL import Image
import io
import cv2

from keras.datasets import mnist

import multiprocessing as mp
from multiprocessing import Pool, Manager, Process
from functools import partial

from . import logging_daily
from . import utils
from keras.utils import to_categorical

######################################################################
# Base Reader
######################################################################
class BaseReader(object):
    """Inherit from this class when implementing new readers."""
    def __init__(self, log, path_info, network_info, verbose=True):
        self.log = log
        self.verbose = verbose
        
        self.data_path = path_info['data_info']['data_path']
        if network_info['model_info']['normalize_sym'] == 'True': self.normalize_sym = True
        else: self.normalize_sym = False
        if network_info['model_info']['n_label'] == 'None': self.n_label = None
        else: self.n_label = int(network_info['model_info']['n_label'])
        if network_info['model_info']['augment'] == 'True': self.augment = True
        else: self.augment = False
        
        self.x_list = None
        self.img_shape = None
        
    def read_dataset(self, data_path):
        raise NotImplementedError()
    
    def get_dataset(self):
        raise NotImplementedError()
    
    def get_cv_index(self, nfold=5):
        raise NotImplementedError()

    def get_augment(self, x):
        for i in range(x.shape[0]):
            if np.random.randint(2, size=1):
                # Flip Horizontally
                if np.random.randint(2, size=1):
                    x[i] = x[i,:,::-1,:] # (N, H, W, C)
                # Channel Noise
                if np.random.randint(2, size=1):
                    if np.random.randint(2, size=1):
                        # uniform noise
                        noise = np.random.uniform(0,0.05,(x.shape[1],x.shape[2]))
                        picked_ch = np.random.randint(3, size=1)[0]
                        x[i,:,:,picked_ch] += noise
                        x[i,:,:,picked_ch] = np.clip(x[i,:,:,picked_ch], a_min=0., a_max=1.)
                    elif np.random.randint(2, size=1):
                        # gray
                        x[i,:,:,:] = np.repeat(np.expand_dims(np.dot(x[i,:,:], [0.299, 0.587, 0.114]), axis=-1), 3, axis=-1)
        return x
    
    def show_class_information(self, y=None):
        if np.any(y == None): y_table = self.y_table
        else: y_table = pd.Series(y)
        y_counts = y_table.value_counts()
        self.log.info('-------------------------------------------------')
        self.log.info('Images per Class')
        self.log.info('\n%s', y_counts)
        self.log.info('-------------------------------------------------')
        self.log.info('Summary')
        self.log.info('\n%s', y_counts.describe())
        self.log.info('-------------------------------------------------')
        
    # def write_embeddings_metadata(self, embedding_metadata_path, e_x, e_y):
    #     with open(embedding_metadata_path,'w') as f:
    #         f.write("Index\tLabel\tClass\n")
    #         for index,label in enumerate(e_y):
    #             f.write("%d\t%s\t%d\n" % (index,"fake",label)) # fake
    #         for index,label in enumerate(e_y):
    #             f.write("%d\t%s\t%d\n" % (len(e_y)+index,"true",10)) # true
                            
    def get_image_shape(self):
        return self.img_shape
    
    def get_cv_index(self, nfold=5, random_state = 12):
        self.log.info('%d-fold Cross Validation Cut' % nfold)
        kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
        return kf.split(range(self.y.shape[0]))
    
    def get_training_validation_index(self, idx, validation_size=0.2):
        return train_test_split(idx, test_size = validation_size)
    
    def get_dataset(self):
        return self.x, self.y
    
    def get_label(self):
        return self.y
    
    def get_n_label(self):
        return self.num_classes
    
    def handle_imbalance(self, train_idx, minarity_group_size = 0.3, minarity_ratio = 0.3, seed=12):
        self.log.info('----------------------------------------------------')
        self.log.info('Handle imbalance')
        self.log.info('Minarity_group_size : %s' % minarity_group_size)
        self.log.info('Minarity_ratio (per group) : %s' % minarity_ratio)
        self.log.info('----------------------------------------------------')
        np.random.seed(seed)
        minarities = np.random.choice(self.y_class, size= int(minarity_group_size * self.y_class.shape[0]))
        pick = []
        if len(minarities) > 0:
            for i, minarity in enumerate(minarities):
                minarity_index = self.y_index.get_loc(minarity)
                delete_size = int(np.sum(minarity_index) * (1-minarity_ratio))
                pick.append(np.random.choice(np.where(minarity_index)[0], replace=False, size=delete_size))
                self.log.info('minarity class - %s : deleted %s of %s' %(minarity, delete_size, np.sum(minarity_index)))
            pick = np.concatenate(pick)
            train_idx = np.setdiff1d(train_idx, pick)
            if self.verbose == True: self.show_class_information(self.y[train_idx])
        return train_idx

    def class_to_categorical(self, y):
        return to_categorical(self.class_to_int(y), self.num_classes)
    
    def categorical_to_series(self, y_coded):
        return pd.Series(np.argmax(y_coded, axis=1)).map(self.y_int_to_class)
    
    def class_to_int(self, y):
        return np.array(pd.Series(y).map(self.y_class_to_int))

    def int_to_class(self, y_int):
        return pd.Series(y_int).map(self.y_int_to_class)

#########################################################################################################
# Toy Sample Reader
#########################################################################################################
class ToyReader(BaseReader):
    def __init__(self, log, path_info, network_info, verbose=True):
        super(ToyReader,self).__init__(log, path_info, network_info, verbose)
        self.read_dataset(nlabel=self.n_label)
        if verbose: self.show_class_information()
        
    def read_dataset(self, nlabel=None):
        dir_path = self.data_path
        self.x = np.load('%s/x.npy'%dir_path).astype(np.float32)
        self.x = self.x.reshape(self.x.shape[0],int(np.sqrt(self.x.shape[1])),int(np.sqrt(self.x.shape[1])),1)
        self.y = np.load('%s/y.npy'%dir_path)
        self.img_shape = self.x.shape[1:]
        
        if not nlabel==None:
            y_table = pd.Series(self.y)
            selected_class = y_table.unique()[:nlabel]
            selected_class = y_table.isin(selected_class)
            self.x_list = self.x[selected_class]
            self.y = self.y[selected_class]
            
        ## to categorical ####################
        self.y_table = pd.Series(self.y)
        self.y_index = pd.Index(self.y)
        self.y_class = np.sort(self.y_table.unique())
        self.y_class_to_int = dict(zip(self.y_class, range(self.y_class.shape[0])))
        self.y_int_to_class = dict(zip(range(self.y_class.shape[0]), self.y_class))
        self.num_classes = len(self.y_class)
        ######################################
        
    def get_batch(self, idxs):
        img_batches = self.x[idxs]
        y = self.class_to_int(self.y[idxs])
        return img_batches, y
    
    def class_to_categorical(self, y):
        return to_categorical(y, self.num_classes)
    
    def categorical_to_class(self, y_coded):
        return np.argmax(y_coded, axis=1)
    
    def except_class(self, train_idx, except_class):
        self.log.info('----------------------------------------------------')
        for unknown_class in except_class:
            self.log.info('Except class %d' % int(unknown_class))
            unknown_class = int(unknown_class)
            train_idx = train_idx[self.y[train_idx]!=unknown_class]
        if self.verbose: self.show_class_information(self.y[train_idx])
        self.log.info('----------------------------------------------------')
        return train_idx
    
#########################################################################################################
# MNIST
#########################################################################################################
class MNISTReader(BaseReader):
    def __init__(self, log, path_info, network_info, mode='train', verbose=True):
        super(MNISTReader,self).__init__(log, path_info, network_info, verbose)
        self.read_dataset(nlabel=self.n_label)
        if verbose: self.show_class_information()
        
    def read_dataset(self, nlabel=None):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x = np.concatenate((x_train, x_test), axis=0)
        self.y = np.concatenate((y_train, y_test), axis=0)
        if not nlabel==None:
            y_table = pd.Series(self.y)
            selected_class = y_table.unique()[:nlabel]
            selected_class = y_table.isin(selected_class)
            self.x_list = self.x[selected_class]
            self.y = self.y[selected_class]
        ## to categorical ####################
        self.y_table = pd.Series(self.y)
        self.y_index = pd.Index(self.y)
        self.y_class = np.sort(self.y_table.unique())
#         self.y_class_to_int = dict(zip(self.y_class, range(self.y_class.shape[0])))
#         self.y_int_to_class = dict(zip(range(self.y_class.shape[0]), self.y_class))
        self.num_classes = len(self.y_class)
        ######################################
        
        # normalize
        if self.normalize_sym:
            # force it to be of shape (...,28,28,1) with range [-1,1]
            self.x = ((self.x - 127.5) / 127.5).astype(np.float32) 
        else:
            self.x = (self.x / 225.).astype(np.float32)
        self.x = np.expand_dims(self.x, axis=-1)
        self.img_shape = self.x.shape[1:]
        
    def get_batch(self, idxs):
        img_batches = self.x[idxs]
        if self.augment:
            img_batches = self.get_augment(img_batches)
#         y = self.class_to_int(self.y[idxs])
        y = self.y[idxs]
        return img_batches, y
    
    def class_to_categorical(self, y):
        return to_categorical(y, self.num_classes)
    
    def categorical_to_class(self, y_coded):
        return np.argmax(y_coded, axis=1)
    
    def except_class(self, train_idx, except_class):
        self.log.info('----------------------------------------------------')
        for unknown_class in except_class:
            self.log.info('Except class %d' % int(unknown_class))
            unknown_class = int(unknown_class)
            train_idx = train_idx[self.y[train_idx]!=unknown_class]
        if self.verbose: self.show_class_information(self.y[train_idx])
        self.log.info('----------------------------------------------------')
        return train_idx
    
#########################################################################################################
# Omniglot
#########################################################################################################
class OmniglotReader(BaseReader):
    def __init__(self, log, path_info, network_info, mode='train', verbose=True):
        super(OmniglotReader,self).__init__(log, path_info, network_info, verbose)
        self.mode = mode
        self.read_dataset(nlabel=self.n_label)
        if verbose: self.show_class_information()

    def read_dataset(self, nlabel=None):
        self.log.info('-------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-------------------------------------------------')
        self.log.info('Loading Omniglot dataset information')
        
        self.img_shape = (105,105,1)
        if self.mode=='train': data_type = 'images_background'
        elif self.mode=='train_small1': data_type = 'images_background_small1'
        elif self.mode=='train_small2': data_type = 'images_background_small2'
        else: data_type = 'images_evaluation'
        self.x_list = np.load('%s/%s_x_list.npy' % (self.data_path, data_type))
        self.y = np.load('%s/%s_y.npy' % (self.data_path, data_type))
        
        if not nlabel==None:
            y_table = pd.Series(self.y)
            selected_class = y_table.unique()[:nlabel]
            selected_class = y_table.isin(selected_class)
            self.x_list = self.x_list[selected_class]
            self.y = self.y[selected_class]
        # else:
        #     y_table = pd.Series(self.y)
        #     y_counts = y_table.value_counts()
        #     selected_class = y_counts[y_counts >= 5].keys()
        #     selected_class = y_table.isin(selected_class)
        #     self.x_list = self.x_list[selected_class]
        #     self.y = self.y[selected_class]
        #     self.not_used_class = y_counts[y_counts < 5].keys()
        
        ## to categorical ####################
        self.y_table = pd.Series(self.y)
        self.y_index = pd.Index(self.y)
        self.y_class = np.sort(self.y_table.unique())
        self.y_class_to_int = dict(zip(self.y_class, range(self.y_class.shape[0])))
        self.y_int_to_class = dict(zip(range(self.y_class.shape[0]), self.y_class))
        self.num_classes = len(self.y_class)
        ######################################
        
        self.y_alphabet = np.array([xpath.split('/')[-3] for xpath in self.x_list])
        
    # TODO except class list...
    # def except_class(self, train_idx, unknown_class='9'):
        # train_idx = np.array(train_idx)
        # return train_idx[self.y[train_idx]!=unknown_class]
    
    def get_cv_index(self, nfold=5, random_state = 12):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        self.log.info('Stratified %d-fold Cross Validation Cut' % nfold)
        kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_state)
        return kf.split(range(self.y.shape[0]), self.y)
    
    def get_dataset(self):
        return self.x_list, self.y
    
    def get_y_alphabet_class(self):
        return self.y_alphabet
    
    def get_label_name(self):
        return np.array(self.y_class)
    
    def get_batch(self, idxs):
        try:
            batch_imgs = []
            batch_idxs = []
            for i in idxs:
                try:
                    batch_imgs.append(self._read_omniglot_image(self.x_list[i]))
                    batch_idxs.append(i)
                except Exception as e:
                    raise ValueError(e)
            batch_imgs = np.array(batch_imgs)
            batch_idxs = np.array(batch_idxs)
            # if self.augment and np.random.choice([0,1], 1, replace=False, p=[0.8,0.2]):
            if self.augment:
                batch_imgs = self.get_augment(batch_imgs)
            if self.normalize_sym:
                batch_imgs = (batch_imgs - 0.5) * 2.
            y = self.class_to_int(self.y[np.array(batch_idxs)])
            return batch_imgs, y
        except Exception as e:
            raise ValueError(e)
    
    def _read_omniglot_image(self, filename):
        try:
            im = Image.open(filename)
#             target_shape = np.array([self.img_shape[1],self.img_shape[0]])
#             im = im.resize(target_shape, PIL.Image.ANTIALIAS)
            im = np.expand_dims((1.-np.array(im).astype(np.float32)), -1)
            # dilation (thickness)
            # kernel = np.ones((3,3),np.uint8)
            # im = np.expand_dims(cv2.dilate(im,kernel,iterations = 1), -1)
            return im
        except Exception as e:
            raise ValueError('Error with %s : %s' % (filename, e))
            # sys.exit()
    
    #########################################################################
    # funtions for augmentation
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    #########################################################################
    def _dilation(self, im, kernal_size=(2,2)):
        # Dilation (thickness)
        kernel = np.ones(kernal_size,np.uint8)
        im = cv2.dilate(im,kernel,iterations = 1)
        im[im>=0.5] = 1.
        im[im<0.5] = 0.
        return np.expand_dims(np.array(im).astype(np.float32), -1)
    
    def _rotation(self, im, max_angle = 10):
        # Rotation
        rows,cols,ch = im.shape
        angle = np.random.choice(np.append(np.arange(-max_angle,max_angle,max_angle//4),max_angle), 1)[0]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        im = cv2.warpAffine(im,M,(cols,rows))
        im[im>=0.5] = 1.
        im[im<0.5] = 0.
        return np.expand_dims(np.array(im).astype(np.float32), -1)
    
    def _affine(self, im, max_tiltrate = 6):
        # Affine transformation
        rows,cols,ch = im.shape
        tiltsize=np.random.choice(np.arange(max_tiltrate//4,max_tiltrate,max_tiltrate//4), 1)[0]
        pts1 = np.float32([[tiltsize,tiltsize],[rows-tiltsize,tiltsize],[tiltsize,cols-tiltsize]])
        pts2 = np.float32([[tiltsize,tiltsize],[rows,0],[0,cols]])
        M = cv2.getAffineTransform(pts1,pts2)
        im = cv2.warpAffine(im,M,(cols,rows))
        im[im>=0.5] = 1.
        im[im<0.5] = 0.
        return np.expand_dims(np.array(im).astype(np.float32), -1)
    
    def _perspective(self, im, max_padsize=6):
        # Perspective tranformation
        rows,cols,ch = im.shape
        padsize=np.random.choice(np.arange(max_padsize//4,max_padsize,max_padsize//4), 1)[0]
        pts1 = np.float32([[padsize,padsize],[rows-padsize,padsize],[padsize,cols-padsize],[rows-padsize,cols-padsize]])
        pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        im = cv2.warpPerspective(im,M,(rows,cols))
        im[im>=0.5] = 1.
        im[im<0.5] = 0.
        return np.expand_dims(np.array(im).astype(np.float32), -1)
        
    def get_augment(self, x):
        for i in range(x.shape[0]):
            if np.random.randint(2, size=1):
#                 if np.random.randint(2, size=1): x[i] = self._dilation(x[i]) # Dilation (thickness)
                if np.random.randint(2, size=1): x[i] = self._rotation(x[i]) # Rotation
                if np.random.randint(2, size=1): x[i] = self._affine(x[i]) # Affine transformation
                if np.random.randint(2, size=1): x[i] = self._perspective(x[i]) # Perspective tranformation
        return x
    
#########################################################################################################
# CelebA
#########################################################################################################
class CelebAReader(BaseReader):
    def __init__(self, log, path_info, network_info, verbose=True):
        super(CelebAReader,self).__init__(log, path_info, network_info, verbose)
        
        self.crop_style=network_info['model_info']['crop_style'].strip()
        self.attr_label=network_info['model_info']['attr_label'].strip()

        self.read_dataset(self.attr_label)
        if verbose: self.show_class_information()

    def read_dataset(self, attr_label='Male'):
        self.log.info('-------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-------------------------------------------------')
        self.log.info('Loading CelebA dataset information')
        
        self.img_shape = (64, 64, 3)
        
        # num_samples = len(os.listdir(self.data_path)) #202599
        # self.datapoint_ids = np.arange(1, num_samples + 1)
        # np.random.shuffle(self.datapoint_ids)
        # self.x_list = ['%.6d.jpg' % i for i in self.datapoint_ids]
        self.x_list = np.load('%s/x_list.npy' % ('/'.join(self.data_path.split('/')[:-1], datatype)))
        
        self.attr = pd.read_csv('/'.join(self.data_path.split('/')[:-1])+'/list_attr_celeba.csv')
        sorterIndex = dict(zip(self.x_list,range(len(self.x_list))))
        self.attr['index'] = self.attr['image_id'].map(sorterIndex)
        self.attr = self.attr.sort_values('index')
        self.y = np.array(self.attr[attr_label])
        self.y[self.y == -1] = 0
        self.class_name = np.array(['no_%s' % attr_label,attr_label])
        self.y_table = pd.Series(self.y)
        self.y_counts = self.y_table.value_counts()
        self.y_index = pd.Index(self.y)
        self.y_class = np.sort(self.y_table.unique())
        self.num_classes = self.y_class.shape[0]
        
    def get_dataset(self):
        return self.x_list, self.y
    
    def get_label_name(self):
        return self.class_name
    
    def get_batch(self, idxs):
        img_batches = np.array([self._read_celeba_image(self.x_list[i]) for i in idxs])
        if self.augment:
            img_batches = self.get_augment(img_batches)
        if self.normalize_sym:
            img_batches = (img_batches - 0.5) * 2.
        return img_batches, self.y[np.array(idxs)]
    
    def _read_celeba_image(self, filename):
        # from WAE
        width = 178
        height = 218
        new_width = 140
        new_height = 140
        im = Image.open(utils.o_gfile((self.data_path, filename), 'rb'))
        if self.crop_style == 'closecrop':
            # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
            left = (width - new_width) / 2.
            top = (height - new_height) / 2.
            right = (width + new_width) / 2.
            bottom = (height + new_height)/2.
            im = im.crop((left, top, right, bottom))
            im = im.resize((64, 64), PIL.Image.ANTIALIAS)
        elif self.crop_style == 'resizecrop':
            # This method was used in ALI, AGE, ...
            im = im.resize((64, 64+14), PIL.Image.ANTIALIAS)
            im = im.crop((0, 7, 64, 64 + 7))
        else:
            raise Exception('Unknown crop style specified')
        return np.array(im).reshape(64, 64, 3) / 255.

#########################################################################################################
# VGG 2 Face
#########################################################################################################
class VGGFace2Reader(BaseReader):
    def __init__(self, log, path_info, network_info, mode='train', verbose=True):
        super(VGGFace2Reader,self).__init__(log, path_info, network_info, verbose)
        
        self.crop_style=network_info['model_info']['crop_style'].strip()
        self.img_shape = np.array([int(ishape.strip()) for ishape in network_info['model_info']['img_shape'].split(',')])
        
        self.mode = mode
        
        self.read_dataset(nlabel=self.n_label)
        if verbose: self.show_class_information()
            
        try: self.feature_b = 'true' == network_info['model_info']['feature_b'].strip().lower()
        except: self.feature_b = False
        if self.feature_b:
            if self.mode == 'train': 
                self.all_features_for_b = np.load('%s/all_features_normalized.npy' % path_info['data_info']['data_path'])
            else:
                self.all_features_for_b = np.load('%s/all_features_of_unknown_normalized.npy' % path_info['data_info']['data_path'])
            self.log.info('Load all features for b: %s' % np.array(len(self.all_features_for_b)))
                    
        try: self.fixed_b_path = network_info['training_info']['fixed_b_path'].strip()
        except: self.fixed_b_path = None
        if self.fixed_b_path is not None: 
            self.all_b = np.load(self.fixed_b_path)
            self.log.info('Load all b: %s' % np.array(self.all_b.shape))

    def read_dataset(self, nlabel=None):
        self.log.info('-------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-------------------------------------------------')
        self.log.info('Loading VGG Face 2 dataset information')
        
        self.log.info('Set image shape : %s' %  self.img_shape)
        
        # names = os.listdir(self.data_path+'/npy_128')
        # if not npersion==None:
        #     names = names[:npersion]
        # file_dict = {}
        # file_dict.update([(name, os.listdir(self.data_path+'/images/%s' % name)) for name in names])
        # self.x_list = np.concatenate([['%s/%s'%(name, path) for path in paths] for name, paths in file_dict.items()])
        # self.y = np.concatenate([[name]*len(paths) for name, paths in file_dict.items()])
        
        if self.mode == 'train': list_path = "%s/%s" % (self.data_path, 'train_list.txt')
        else: list_path = "%s/%s" % (self.data_path, 'test_list.txt')
        with open(list_path, 'r') as f:
            self.x_list = f.read()
            self.x_list = np.array(self.x_list.split('\n')[:-1])
        x_table = pd.Series(self.x_list)
        self.y = np.array(x_table.map(lambda x : x.split("/")[0]))
        
        if not nlabel==None:
            y_table = pd.Series(self.y)
            # selected_class = y_table.unique()[np.random.choice(np.arange(y_table.unique().shape[0]), nlabel)]
            selected_class = np.sort(y_table.unique())[:nlabel]
            selected_class = y_table.isin(selected_class)
            self.x_list = self.x_list[selected_class]
            self.y = self.y[selected_class]
        
        ## to categorical ####################
        self.y_table = pd.Series(self.y)
        self.y_index = pd.Index(self.y)
        self.y_class = np.sort(self.y_table.unique())
        self.y_class_to_int = dict(zip(self.y_class, range(self.y_class.shape[0])))
        self.y_int_to_class = dict(zip(range(self.y_class.shape[0]), self.y_class))
        self.num_classes = len(self.y_class)
        ######################################
        
        if self.mode == 'train': self.image_info = pd.read_csv('%s/%s' % (self.data_path, 'bb_landmark/loose_bb_train.csv'))
        else: self.image_info = pd.read_csv('%s/%s' % (self.data_path, 'bb_landmark/loose_bb_test.csv'))
        self.image_info = self.image_info.set_index(['NAME_ID'])
            
    def except_class(self, except_class):
        self.log.info('----------------------------------------------------')
        train_idx = np.arange(self.y.shape[0])
        for unknown_class in except_class:
            self.log.info('Except class %s' % unknown_class)
            train_idx = train_idx[self.y[train_idx]!=unknown_class]
        self.x_list = self.x_list[train_idx]
        self.y = self.y[train_idx]
        if self.verbose: self.show_class_information(self.y)
        self.log.info('----------------------------------------------------')
        
        ## to categorical ####################
        self.y_table = pd.Series(self.y)
        self.y_index = pd.Index(self.y)
        self.y_class = self.y_table.unique()
        self.y_class_to_int = dict(zip(self.y_class, range(self.y_class.shape[0])))
        self.y_int_to_class = dict(zip(range(self.y_class.shape[0]), self.y_class))
        self.num_classes = len(self.y_class)
        ######################################
    
    def get_dataset(self):
        return self.x_list, self.y
    
    def get_label_name(self):
        return np.array(self.y_class)
    
    def get_batch(self, idxs):
        try:
            batch_imgs = []
            batch_idxs = []
            for i in idxs:
                try:
                    batch_imgs.append(self._read_vgg2face_image(self.x_list[i]))
                    batch_idxs.append(i)
                except Exception as e:
                    raise ValueError(e)
            batch_imgs = np.array(batch_imgs)
            batch_idxs = np.array(batch_idxs)
            if self.augment:
                batch_imgs = self.get_augment(batch_imgs)
            if self.normalize_sym:
                batch_imgs = (batch_imgs - 0.5) * 2.
            y = self.class_to_int(self.y[np.array(batch_idxs)])

            if self.feature_b:
                batch_feature_b = self.all_features_for_b[idxs]
                if self.fixed_b_path is not None: ## TODO: not using feature_b
                    batch_b = self.all_b[idxs]
                    return [batch_imgs, batch_feature_b, batch_b], y
                else: return [batch_imgs, batch_feature_b], y
            else:
                return batch_imgs, y
        except Exception as e:
            raise ValueError(e)
    
    def _read_vgg2face_image(self, filename):
        try:
            if self.mode == 'train': path = '%s/%s/%s' % (self.data_path, 'train', filename)
            else: path = '%s/%s/%s' % (self.data_path, 'test', filename)
#             path = '%s/%s/%s' % (self.data_path, 'test', filename)
            # im = cv2.imread(path)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # im = Image.fromarray(im)
            # with open(path) as f:
            #     im = io.BytesIO(f.read())
            # im = Image.open(im)
            im = Image.open(path)
            if self.crop_style == 'closecrop':
                # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
                ## TODO : fix
                # info = self.image_info[self.image_info['NAME_ID'] ==  filename.split('.')[0]]
                info = self.image_info.loc[filename.split('.')[0]]
                # positon = (left, top, right, bottom)
                # bd = 32
                bd = int(max(info['W'] * 0.2, info['H'] * 0.2))
                im = im.crop((max(0,info['X']-bd),
                              max(0,info['Y']-bd), 
                              min(info['X']+info['W']+bd,im.size[0]),
                              min(info['Y']+info['H']+bd,im.size[1])))
                
                #im = im.resize(self.img_shape[:2], PIL.Image.ANTIALIAS)
                # bigger = self.img_shape[0] + bd*2
                if np.any(im.size != self.img_shape[:2]):
                    # resize_shape = np.array([bigger]*2)
                    resize_shape = np.array(self.img_shape[:2])
                    resize_shape[np.argmax(im.size)] = int(resize_shape[0]*np.max(im.size)/np.min(im.size))
                    im = im.resize(resize_shape, PIL.Image.ANTIALIAS)
                    border = 0.5*(resize_shape-self.img_shape[:2]).astype(np.int)
                    ## TODO : augment fix
                    # zitter = int(np.random.choice(np.arange(-np.max(border)*0.25,np.max(border)*0.25),1)[0])
                    zitter = 0
                    im = im.crop((border[0], border[1]+zitter, self.img_shape[0]+border[0], self.img_shape[1]+border[1]+zitter))
            elif self.crop_style == 'resizecrop':
                # This method was used in ALI, AGE, ...
                im = im.resize((self.img_shape[0], self.img_shape[1]+14), PIL.Image.ANTIALIAS)
                im = im.crop((0, 7, self.img_shape[0], self.img_shape[1] + 7))
            else:
                raise Exception('Unknown crop style specified')
            im = np.array(im.convert("RGB")).reshape(self.img_shape[0], self.img_shape[1], 3) / 255.
            return im
        except Exception as e:
            raise ValueError('Error with %s : %s' % (path, e))
            # sys.exit()
            
    def get_augment(self, x):
        for i in range(x.shape[0]):
            if np.random.randint(2, size=1):
                # Channel Noise
                if np.random.randint(2, size=1):
                    # uniform noise
                    noise = np.random.uniform(0,0.05,(x.shape[1],x.shape[2]))
                    picked_ch = np.random.randint(3, size=1)[0]
                    x[i,:,:,picked_ch] += noise
                    x[i,:,:,picked_ch] = np.clip(x[i,:,:,picked_ch], a_min=0., a_max=1.)
                elif np.random.randint(2, size=1):
                    # gray
                    x[i,:,:,:] = np.repeat(np.expand_dims(np.dot(x[i,:,:], [0.299, 0.587, 0.114]), axis=-1), 3, axis=-1)
                elif np.random.randint(2, size=1):
                    x[i,:,:,:] = x[i,:,::-1,:]
        return x
         
#########################################################################################################
# Whale's tail
#########################################################################################################
class WhaleReader(BaseReader):
    def __init__(self, log, path_info, network_info, verbose=True):
        super(WhaleReader,self).__init__(log, path_info, network_info, verbose)
        
        self.crop_style=network_info['model_info']['crop_style'].strip()
        self.read_dataset(nlabel=self.n_label)
        if verbose: self.show_class_information()

    def read_dataset(self, nlabel=None):
        self.log.info('-------------------------------------------------')
        self.log.info('Construct Dataset')
        self.log.info('-------------------------------------------------')
        self.log.info('Loading Whale dataset information')
        
        self.img_shape = (64,128,3)
        
#         with open("%s/%s" % (self.data_path, 'train_list.txt'), 'r') as f:
#             self.x_list = f.read()
#             self.x_list = np.array(self.x_list.split('\n')[:-1])
#         x_table = pd.Series(self.x_list)
#         self.y = np.array(x_table.map(lambda x : x.split("/")[0]))

        df = pd.read_csv('%s/train.csv' % self.data_path).set_index('Image')
        new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
        train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training

        self.x_list = train_df.index
        self.y = train_df.Id.values
        
        if not nlabel==None:
            y_table = pd.Series(self.y)
            selected_class = y_table.unique()[:nlabel]
            selected_class = y_table.isin(selected_class)
            self.x_list = self.x_list[selected_class]
            self.y = self.y[selected_class]
        # else:
        #     y_table = pd.Series(self.y)
        #     y_counts = y_table.value_counts()
        #     selected_class = y_counts[y_counts >= 5].keys()
        #     selected_class = y_table.isin(selected_class)
        #     self.x_list = self.x_list[selected_class]
        #     self.y = self.y[selected_class]
        #     self.not_used_class = y_counts[y_counts < 5].keys()
                
        ## to categorical ####################
        self.y_table = pd.Series(self.y)
        self.y_index = pd.Index(self.y)
        self.y_class = np.sort(self.y_table.unique())
        self.y_class_to_int = dict(zip(self.y_class, range(self.y_class.shape[0])))
        self.y_int_to_class = dict(zip(range(self.y_class.shape[0]), self.y_class))
        self.num_classes = len(self.y_class)
        ######################################
        
    # TODO except class list...
    # def except_class(self, train_idx, unknown_class='9'):
        # train_idx = np.array(train_idx)
        # return train_idx[self.y[train_idx]!=unknown_class]
    
    def get_dataset(self):
        return self.x_list, self.y
    
    def get_label_name(self):
        return np.array(self.y_class)
    
    def get_batch(self, idxs):
        try:
            batch_imgs = []
            batch_idxs = []
            for i in idxs:
                try:
                    batch_imgs.append(self._read_whale_image(self.x_list[i]))
                    batch_idxs.append(i)
                except Exception as e:
                    raise ValueError(e)
            batch_imgs = np.array(batch_imgs)
            batch_idxs = np.array(batch_idxs)
            if self.augment:
                batch_imgs = self.get_augment(batch_imgs)
            if self.normalize_sym:
                batch_imgs = (batch_imgs - 0.5) * 2.
            y = self.class_to_int(self.y[np.array(batch_idxs)])
            return batch_imgs, y
        except Exception as e:
            raise ValueError(e)
    
    def _read_whale_image(self, filename):
        try:
            path = '%s/%s/%s' % (self.data_path, 'train', filename)
            im = Image.open(path)
            
            target_shape = np.array([self.img_shape[1],self.img_shape[0]])
            if (im.size[1] / im.size[0] < 0.5):
            #     target_2 = (int(im.size[0] * target_shape[1]/im.size[1]), target_shape[1]) # 짧은쪽 맞추고 긴쪽 자르기
                target_2 = target_shape
            else: # 긴쪽 맞추고 짧은쪽 자르기
                target_2 = (target_shape[0],int(im.size[1] * target_shape[0]/im.size[0]))
            bd = (target_2 - target_shape)//2
            bd = np.clip(bd, 0., max(bd))
            
            im = im.resize(target_2, PIL.Image.ANTIALIAS)
            im = im.crop((bd[0], bd[1], target_shape[0]+bd[0], target_shape[1]+bd[1]))
            im = np.array(im.convert("RGB")).reshape(self.img_shape[0], self.img_shape[1], 3) / 255.
            return im
        except Exception as e:
            raise ValueError('Error with %s : %s' % (path, e))
            # sys.exit()
            
#####################################################################################################################
if __name__ == "__main__":
    # Logger
    logger = logging_daily.logging_daily('base_model/config/log_info.yaml')
    logger.reset_logging()
    log = logger.get_logging()
    
    path_info = {'data_info': {'data_path':''}}
    network_info = {'model_info': {'normalize_sym':'False', 'n_label':10, 'augment':'False'}}
    
    #############################################################################################################
    # Test MNISTReader
    #############################################################################################################
    log.info('------------------------------------------------------------------------')
    reader = MNISTReader(log, path_info, network_info, verbose=True)
    log.info('x_train, y_train shape: %s, %s', reader.x_train.shape, reader.y_train.shape)
    log.info('x_test, y_test shape: %s, %s', reader.x_test.shape, reader.y_test.shape)
    
    log.info('handle imbalance')
    reader.handle_imbalance()
    log.info('x_train, y_train shape: %s, %s', reader.x_train.shape, reader.y_train.shape)
    log.info('x_test, y_test shape: %s, %s', reader.x_test.shape, reader.y_test.shape)
    
    log.info('------------------------------------------------------------------------')
