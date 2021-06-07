import os
import sys
import timeit
import numpy as np
import pandas as pd
import copy

from keras.utils import Sequence

from . import logging_daily
from . import readers
from .utils import convert_bytes

#########################################################################################################################
class train_generator(Sequence):
    # keras.utils.Sequence class was used to avoid duplicating data to multiple workers
    # Sampler : instanse of BaseSampler class or BaseSampler inheritted Sampler class
    def __init__(self, sampler):
        self.sampler = sampler
        self.batch_size = sampler.get_batch_size()
        self.steps = sampler.get_steps()
        self.sampler.on_training_start()
        
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        x, y = self.sampler.load_batch(idx)
        return x, y
    
    def on_batch_end(self):
        self.sampler.on_batch_end()
        
    def on_epoch_end(self):
        self.sampler.on_epoch_end()
        
class evaluation_generator(Sequence):
    # keras.utils.Sequence class was used to avoid duplicating data to multiple workers
    # Sampler : instanse of BaseSampler class or BaseSampler inheritted Sampler class
    def __init__(self, sampler):
        self.sampler = sampler
        self.batch_size = sampler.get_batch_size()
        self.steps = sampler.get_steps()
        self.sampler.on_training_start()
            
    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        x, y = self.sampler.load_batch(idx)
        return x, y
    
    def on_batch_end(self):
        self.sampler.on_batch_end()
    
    def on_epoch_end(self):
        self.sampler.on_epoch_end()

class predict_generator(Sequence):
    # thread_safe generator
    def __init__(self, sampler):
        self.sampler = sampler
        self.batch_size = sampler.get_batch_size()
        self.steps = sampler.get_steps()
            
    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
        x, _= self.sampler.load_batch(idx)
        return x
    
#########################################################################################################################

class BaseSampler():
    """
    Probability based Sampler for fit_generator and data generator which inherited keras.utils.Sequence
    Inherit from this class when implementing new probability based sampling.
    
    Example
    =================
    TODO
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        #super(BaseSampler, self).__init__()
        
        self.log = log
        self.train_idx = np.array(train_idx)
        self.reader = reader
        self.parallel = parallel
        self.verbose = verbose
        
        if training_info['sequential'] == 'True': self.sequential = True
        else: self.sequential = False
        if training_info['replace'] == 'True': self.replace = True
        else: self.replace = False
        if training_info['steps_per_epoch'] == 'None': self.steps_per_epoch = None
        else: self.steps_per_epoch = int(training_info['steps_per_epoch'])
        self.batch_size = int(training_info['batch_size'])
        
        y_table = pd.Series(reader.get_label()[self.train_idx])
        self.y_counts = y_table.value_counts()
        self.y_class = y_table.unique()
        self.y_index = pd.Index(y_table)
        
        self.probability = None
        
        if verbose:
            self.log.info('-------------------------------------------------')
            self.log.info('Images per Class')
            self.log.info('\n%s', self.y_counts)
            self.log.info('-------------------------------------------------')
            self.log.info('Summary')
            self.log.info('\n%s', self.y_counts.describe())
            self.log.info('-------------------------------------------------')
        
    def set_probability_vector(self):
        """
        Probability Vector for Probability sampling
        """
        raise NotImplementedError()
        
    def get_proability_vector(self):
        return self.probability
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_steps(self):
        if self.steps_per_epoch == None:
            return np.ceil(self.train_idx.shape[0]/self.batch_size).astype(np.int)
            #return np.floor(self.train_idx.shape[0]/self.batch_size).astype(np.int)
        else:
            return self.steps_per_epoch
    
    def probability_sampling(self, idxs, size, replace=False):
        """
        Probability based Index Sampling Functions
        probability : The probabilities associated with each entry in a. 
                      If not given the sample assumes a uniform distribution over all entries in a.
        """
        try:
            return np.random.choice(idxs, size=size, replace=replace, p=self.probability)
        except Exception as e:
            self.log.error(e)
            raise ValueError(e)
    
    def on_training_start(self):
        if self.sequential: np.random.shuffle(self.train_idx)
        else: self.set_probability_vector()
        
    def on_epoch_end(self):
        if self.sequential: np.random.shuffle(self.train_idx)
        else: self.set_probability_vector()
    
    def on_batch_end(self):
        pass
        
    def load_batch(self, i):
        try:
            if self.sequential:
                idxs = self.train_idx[np.arange(i*self.batch_size,min((i+1)*self.batch_size, len(self.train_idx)))]
            else:
                idxs = self.probability_sampling(self.train_idx, self.batch_size, replace=self.replace)
            return self.reader.get_batch(idxs)
        except Exception as e:
            raise ValueError(e)

            
#########################################################################################################################
## Uniform sampling / Oversampling / 
#########################################################################################################################

class UniformSampler(BaseSampler):
    """
    Sampling from uniform distribution
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(UniformSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        
    def set_probability_vector(self):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform'
        """
        self.probability = np.zeros(self.y_index.shape[0])
        self.probability[:] = 1./self.y_index.shape[0]
        return self.probability
    
    def on_epoch_end(self):
        if self.sequential: np.random.shuffle(self.train_idx)
    
class OverSamplingSampler(BaseSampler):
    """
    Oversampling from minarity class to deal with imbalance
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(OverSamplingSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        self.alpha = 1.
        try: self.decay = float(training_info['sampler_decay'])
        except : self.decay = 1.
            
    def set_probability_vector(self):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'equiprobability'
        """
        ## TODO : decay..
        class_probability = copy.deepcopy(self.y_counts)
#         class_probability[:] = np.reciprocal(class_probability.get_values().astype(np.float))

        oversampling_prob = np.zeros(self.y_index.shape)
        for class_label in self.y_class:
            idx = self.y_index.get_loc(class_label)
            oversampling_prob[idx] = 1./class_probability[class_label]
        oversampling_prob = oversampling_prob / np.sum(oversampling_prob)
        
        uniform_prob = np.ones(self.y_index.shape)
        uniform_prob = uniform_prob/np.sum(uniform_prob)
        
        self.probability = self.alpha*oversampling_prob + (1.- self.alpha)*uniform_prob
        self.probability = np.clip(self.probability, 0.,np.max(self.probability))
        self.probability = self.probability / np.sum(self.probability)
        return self.probability
    
    def on_epoch_end(self):
        ## TODO : consider only self.sequential == False
        if self.sequential: np.random.shuffle(self.train_idx)
        else:
            if self.decay < 1.: self.alpha *= self.decay
            self.set_probability_vector()
            
#########################################################################################################################
class BatchClassSampler(BaseSampler):
    """
    Restrict class per batch
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(BatchClassSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        if self.sequential: raise ValueError('You cannot use sequential=True with BatchClassSampler')
        
        ## TODO : error raise
        self.alpha = 1.
        try: self.class_per_batch = int(training_info['sampler_class_per_batch'])
        except: raise ValueError
        try: self.decay = float(training_info['sampler_decay'])
        except : self.decay = 1.
        
    def set_probability_vector(self):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform over selected class (per epoch)'
        """
        picked_class = np.random.choice(self.y_class, size=self.class_per_batch, replace=False)
        weighted_prob = np.zeros(self.y_index.shape)
        for class_label in picked_class:
            idx = self.y_index.get_loc(class_label)
            weighted_prob[idx] = 1.
        weighted_prob = weighted_prob / np.sum(weighted_prob)
        
        uniform_prob = np.ones(self.y_index.shape)
        uniform_prob = uniform_prob / np.sum(uniform_prob)
        
        self.probability = self.alpha*weighted_prob + (1.- self.alpha)*uniform_prob
        # self.probability = (1.-self.alpha)*weighted_prob + self.alpha*uniform_prob
        self.probability = np.clip(self.probability, 0.,np.max(self.probability))
        self.probability = self.probability / np.sum(self.probability)
        
    def on_batch_end(self):
        self.set_probability_vector()
    
    def on_epoch_end(self):
        ## TODO : consider only self.sequential == False
        if self.sequential: np.random.shuffle(self.train_idx)
        else:
            if self.decay < 1.: self.alpha *= self.decay
            self.set_probability_vector()
    
    def load_batch(self, i):
        try:
            if self.sequential:
                idxs = self.train_idx[np.arange(i*self.batch_size,min((i+1)*self.batch_size, len(self.train_idx)))]
            else:
                idxs = self.probability_sampling(self.train_idx, self.batch_size, replace=self.replace)
            self.on_batch_end()
            return self.reader.get_batch(idxs)
        except Exception as e:
            raise ValueError(e)

class BatchClassImportanceSampler(BaseSampler):
    """
    Restrict class per batch
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(BatchClassImportanceSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        
        ## TODO : error raise
        self.alpha = 1.
        try: self.class_per_batch = int(training_info['sampler_class_per_batch'])
        except: raise ValueError
        try: self.decay = float(training_info['sampler_decay'])
        except : self.decay = 1.
        
    def set_probability_vector(self):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform over selected class (per epoch)'
        """
        picked_class = np.random.choice(self.y_class, size=self.class_per_batch, replace=False)
        
        class_probability = copy.deepcopy(self.y_counts)
#         class_probability[np.setdiff1d(self.y_class, picked_class)] = 0.
#         class_probability[picked_class] = np.reciprocal(class_probability[picked_class].get_values().astype(np.float))     
        
        weighted_prob = np.zeros(self.y_index.shape)
        for class_label in picked_class:
            idx = self.y_index.get_loc(class_label)
            weighted_prob[idx] = 1./class_probability[class_label]
        weighted_prob = weighted_prob / np.sum(weighted_prob)
        
        uniform_prob = np.ones(self.y_index.shape)
        uniform_prob = uniform_prob / np.sum(uniform_prob)
        
        self.probability = self.alpha*weighted_prob + (1.- self.alpha)*uniform_prob
        # self.probability = (1.-self.alpha)*weighted_prob + self.alpha*uniform_prob
        self.probability = np.clip(self.probability, 0.,np.max(self.probability))
        self.probability = self.probability / np.sum(self.probability)
        
    def on_batch_end(self):
        self.set_probability_vector()
    
    def on_epoch_end(self):
        ## TODO : consider only self.sequential == False
        if self.sequential: np.random.shuffle(self.train_idx)
        else:
            if self.decay < 1.: self.alpha *= self.decay
            self.set_probability_vector()

#########################################################################################################################
class EpochClassSampler(BaseSampler):
    """
    Restrict class per Epoch
    """
    def __init__(self, log, train_idx, reader, training_info, parallel=0, verbose=True):
        super(EpochClassSampler, self).__init__(log, train_idx, reader, training_info, parallel, verbose)
        if self.sequential: raise ValueError('You cannot use sequential=True with BatchClassSampler')
        
        ## TODO : error raise
        self.alpha = 1.
        try: self.class_per_epoch = int(training_info['sampler_class_per_epoch'])
        except: raise ValueError
        try: self.class_per_batch = int(training_info['sampler_class_per_batch'])
        except: self.class_per_batch= None
        try: self.decay = float(training_info['sampler_decay'])
        except : self.decay = 1.
        self.picked_class_per_epoch = np.random.choice(self.y_class, size=self.class_per_epoch, replace=False)
        
    def set_probability_vector(self):
        """
        Probability Vector for Probability sampling
        sampling_distribution : 'uniform over selected class (per epoch)'
        """
        if self.class_per_batch != None:
            picked_class = np.random.choice(self.picked_class_per_epoch, size=self.class_per_batch, replace=False)
        else:
            picked_class = self.picked_class_per_epoch
        weighted_prob = np.zeros(self.y_index.shape)
        for class_label in picked_class:
            idx = self.y_index.get_loc(class_label)
            weighted_prob[idx] = 1.
        weighted_prob = weighted_prob / np.sum(weighted_prob)
        
        uniform_prob = np.ones(self.y_index.shape)
        uniform_prob = uniform_prob / np.sum(uniform_prob)
        
        self.probability = self.alpha*weighted_prob + (1.- self.alpha)*uniform_prob
        # self.probability = (1.-self.alpha)*weighted_prob + self.alpha*uniform_prob
        self.probability = np.clip(self.probability, 0.,np.max(self.probability))
        self.probability = self.probability / np.sum(self.probability)
        
    def on_batch_end(self):
        self.set_probability_vector()
    
    def on_epoch_end(self):
        ## TODO : consider only self.sequential == False
        if self.sequential: np.random.shuffle(self.train_idx)
        else:
            if self.decay < 1.: 
                self.alpha *= self.decay
                self.class_per_epoch = min(len(self.y_class), int(self.class_per_epoch*(2.-self.decay)))
            self.picked_class_per_epoch = np.random.choice(self.y_class, size=self.class_per_epoch, replace=False)
            self.set_probability_vector()